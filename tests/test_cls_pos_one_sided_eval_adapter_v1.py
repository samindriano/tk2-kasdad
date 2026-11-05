"""Synthetic integration checks for the additive evaluator binding adapter."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from scripts import cls_pos_one_sided_eval_adapter_v1 as ADAPTER
from scripts import evaluate_optimization_v3_cls_pos_frequency_ridge_v1 as OLD_EVAL
from scripts import evaluate_optimization_v3_cls_pos_one_sided_ridge_v1 as EVAL
from scripts import optimization_v3_cls_pos_one_sided_ridge_v1 as RUNNER
from src.models.pos_one_sided_ridge_v1 import ARMS, ORIGINS, route_probabilities


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=False)
    if proc.returncode:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _write_train(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    cycles = np.arange(47, 80, dtype=np.int64)
    ids = np.asarray([f"ADAPTER-SYN-{index:03d}" for index in range(len(cycles))])
    labels = (np.arange(len(cycles)) % 3 == 0).astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "cycle_code", "artifact_signal", "signal_stream"])
        for record_id, cycle, label in zip(ids, cycles, labels):
            writer.writerow([record_id, int(cycle), int(label), "synthetic-only"])
    return pd.DataFrame({"record_id": ids, "cycle_code": cycles}), labels


def _git_init(repo: Path) -> None:
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Synthetic Adapter QA")
    _git(repo, "config", "user.email", "synthetic-adapter@example.invalid")


def run_synthetic_prepare_entry_qa() -> dict[str, Any]:
    """Run actual prepare_predictions on temporary synthetic external fixtures.

    The campaign evaluator, sparse matrix operations, bag-logit replay,
    probability routing, prediction archive and manifest writer remain real.
    Only boundary inputs are supplied by deterministic synthetic fixtures.
    """
    existing_binding = getattr(EVAL, "ARMS", None)
    if existing_binding is not None and tuple(existing_binding) != tuple(ARMS):
        raise AssertionError("Frozen evaluator has a conflicting ARMS binding before adapter install")
    binding = ADAPTER.install_frozen_arm_binding()
    if tuple(EVAL.ARMS) != tuple(ARMS):
        raise AssertionError("Adapter did not bind the model's exact arm tuple")

    with tempfile.TemporaryDirectory(prefix="cls_eval_adapter_qa_") as temp_name:
        repo = Path(temp_name)
        _git_init(repo)
        train_rel = Path("data/raw/classification/Train.csv")
        train_path = repo / train_rel
        identity, labels = _write_train(train_path)
        ids = identity["record_id"].astype(str).to_numpy()
        cycles = identity["cycle_code"].to_numpy(dtype=np.int64)
        train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()

        # Artifacts referenced by the unchanged evaluation loop are synthetic
        # stubs. Their content-address checks remain active.
        source_freeze_rel = RUNNER.SOURCE_FREEZE_REL
        root_pass_rel = RUNNER.ROOT_SOURCE_PASS_REL
        preflight_rel = RUNNER.SOURCE_PREFLIGHT_REL
        for relative, contents in ((source_freeze_rel, b"synthetic-source-freeze"),
                                   (root_pass_rel, b"synthetic-root-pass"),
                                   (preflight_rel, b"synthetic-preflight")):
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)

        a02: dict[str, np.ndarray] = {}
        caches: dict[int, dict[str, Any]] = {}
        matrices: dict[int, dict[str, sparse.csr_matrix]] = {}
        accounting = {"fit_units": []}
        fake_bag = type("SyntheticBag", (), {})()
        fake_bag.classes_ = np.array([0, 1], dtype=np.int8)
        fake_bag.coef_ = np.zeros((1, 2), dtype=np.float64)
        fake_bag.intercept_ = np.zeros(1, dtype=np.float64)

        for origin in ORIGINS:
            rows = np.flatnonzero(cycles > origin).astype(np.int64)
            q_ids = ids[rows]
            q_cycles = cycles[rows]
            present = (np.arange(len(rows)) % 4) != 0
            missing = ~present
            p_a = np.linspace(0.21, 0.79, len(rows), dtype=np.float64)
            p_hgb = np.linspace(0.17, 0.43, len(rows), dtype=np.float64)
            p_uniform = np.full(len(rows), 0.5, dtype=np.float64)
            p_uniform[missing] = p_hgb[missing]
            entry = np.where(np.arange(len(rows)) % 2, "low", "high")
            text_load = np.asarray([f"Q{1 + (index % 4)}" for index in range(len(rows))])
            a02.update({
                f"E{origin}_rows": rows,
                f"E{origin}_ids": q_ids,
                f"E{origin}_cycles": q_cycles,
                f"E{origin}_missing": missing,
                f"E{origin}_pA": p_a,
                f"E{origin}_pP": p_uniform,
                f"E{origin}_group_availability": np.where(missing, "missing", "present"),
                f"E{origin}_group_entry_friction": entry,
                f"E{origin}_group_text_load": text_load,
            })
            score_base = sparse.csr_matrix((len(rows), 2), dtype=np.float64)
            position = sparse.csr_matrix(np.column_stack([
                np.ones(len(rows)), np.arange(len(rows), dtype=np.float64) / 10.0,
                (np.arange(len(rows), dtype=np.float64) % 3) / 5.0,
            ]))
            score_augmented = sparse.hstack([score_base, position], format="csr")
            matrices[origin] = {"score_base": score_base, "score_position": score_augmented}

            matrix_manifest = {
                "matrices": {
                    "score_base": {"sha256": f"synthetic-base-{origin}"},
                    "score_position": {"sha256": f"synthetic-position-{origin}"},
                }
            }
            bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
            bag_rel = f"{OLD_EVAL.PILOT_ROOT}/units/{bag_id}.joblib"
            bag_receipt_rel = f"{OLD_EVAL.PILOT_ROOT}/receipts/{bag_id}.json"
            bag_path, bag_receipt_path = repo / bag_rel, repo / bag_receipt_rel
            bag_path.parent.mkdir(parents=True, exist_ok=True)
            bag_receipt_path.parent.mkdir(parents=True, exist_ok=True)
            bag_path.write_bytes(f"synthetic bag {origin}".encode())
            bag_receipt_path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
            accounting["fit_units"].append({
                "fit_id": bag_id,
                "artifact_sha256": hashlib.sha256(bag_path.read_bytes()).hexdigest(),
                "receipt_sha256": hashlib.sha256(bag_receipt_path.read_bytes()).hexdigest(),
            })
            manifest_path = repo / OLD_EVAL.PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"origin": origin}), encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            caches[origin] = {
                "rows": rows, "ids": q_ids, "cycles": q_cycles,
                "pos": {"prob_missing_hgb": p_hgb},
                "pilot": {"prob_bag": np.full(len(rows), 0.5, dtype=np.float64)},
                "manifest": matrix_manifest,
            }

        def synthetic_path(relative: str) -> Path:
            return repo / relative

        def fake_read_json(file_path: Path) -> dict[str, Any]:
            if file_path == synthetic_path(OLD_EVAL.PILOT_ACCOUNTING_REL):
                return accounting
            return json.loads(file_path.read_text(encoding="utf-8"))

        def fake_identity_cache(origin: int, _identity: pd.DataFrame,
                                _a02: dict[str, np.ndarray]) -> dict[str, Any]:
            return caches[origin]

        def fake_gamma(origin: int, arm: str):
            gamma = np.zeros(3, dtype=np.float64) if arm == "UNIFORM" else np.full(
                3, 0.01 if arm == "ONE_SIDED" else 0.02, dtype=np.float64)
            return gamma, {"fit_id": f"SYN-{origin}-{arm}", "gamma_sha256": "synthetic"}

        def fake_load_matrix(manifest: dict[str, Any], name: str) -> sparse.csr_matrix:
            origin = 67 if manifest is caches[67]["manifest"] else 75
            return matrices[origin][name]

        freeze_stub = {
            "source_commit": "synthetic-parent-commit",
            "source_hashes": {"synthetic": "source-hash"},
            "input_hashes": {"synthetic": "input-hash"},
        }
        resource_stub = {"policy": "synthetic-saved-replay", "initial_ram_pass": True}
        pred_rel = "synthetic/runs/predictions.npz"
        manifest_rel = "synthetic/reports/PREDICTIONS_MANIFEST.json"

        with (
            patch.object(EVAL, "path", side_effect=synthetic_path),
            patch.object(EVAL, "PREDICTION_REL", pred_rel),
            patch.object(EVAL, "PREDICTION_MANIFEST_REL", manifest_rel),
            patch.object(EVAL, "read_json", side_effect=fake_read_json),
            patch.object(EVAL, "_verify_fit_ledger", return_value={"fit_ids_in_order": list(RUNNER.FIT_IDS)}),
            patch.object(EVAL, "_read_identity_and_cache", side_effect=fake_identity_cache),
            patch.object(EVAL, "_load_gamma", side_effect=fake_gamma),
            patch.object(EVAL.joblib, "load", return_value={"estimator": fake_bag}),
            patch.object(OLD_EVAL, "safe_npz_load", return_value=a02),
            patch.object(OLD_EVAL, "load_matrix", side_effect=fake_load_matrix),
            patch.object(OLD_EVAL, "gamma_from_artifact", return_value=(
                np.zeros(3, dtype=np.float64), {"gamma_sha256": "synthetic-uniform"})),
            patch.object(OLD_EVAL, "SharedJobLock", return_value=contextlib.nullcontext()),
            patch.object(RUNNER, "verify_source_freeze", return_value=freeze_stub),
            patch.object(RUNNER, "require_root_source_pass", return_value={"status": "ROOT_SOURCE_PASS"}),
            patch.object(RUNNER, "check_official_identities", return_value={"target_values_opened": False}),
            patch.object(RUNNER, "_read_identity_frame", return_value=identity),
            patch.object(RUNNER, "require_resource", return_value=resource_stub),
            patch.object(RUNNER, "require_actual_boundary", return_value=resource_stub),
        ):
            manifest = EVAL.prepare_predictions()

        prediction_path = repo / pred_rel
        manifest_path = repo / manifest_rel
        with np.load(prediction_path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        if any(key.endswith("_y") or "artifact_signal" in key.lower() for key in arrays):
            raise AssertionError("Synthetic target-free prediction bundle contains labels")
        if (manifest.get("query_target_labels_read") is not False
                or manifest.get("target_npz_members_requested") != []
                or manifest.get("estimator_fits") != 0):
            raise AssertionError("Synthetic prediction manifest did not preserve target-free flags")

        # Commit raw Train and prediction archive first; the actual label guard
        # must reject an uncommitted prediction manifest before target loading.
        _git(repo, "add", str(train_rel), pred_rel)
        _git(repo, "commit", "--quiet", "-m", "synthetic prediction bundle before labels")
        try:
            EVAL.read_query_labels_after_manifest_commit(
                manifest, arrays, repo_root=repo, train_path=train_path,
                prediction_path=prediction_path, manifest_rel=manifest_rel,
                expected_train_sha256=train_sha, expected_train_rows=len(identity),
                expected_cycle_bounds=(47, 79),
            )
        except EVAL.OneSidedEvaluationError:
            pass
        else:
            raise AssertionError("Uncommitted synthetic prediction manifest did not block labels")
        _git(repo, "add", manifest_rel)
        _git(repo, "commit", "--quiet", "-m", "commit synthetic prediction manifest")
        query_labels = EVAL.read_query_labels_after_manifest_commit(
            manifest, arrays, repo_root=repo, train_path=train_path,
            prediction_path=prediction_path, manifest_rel=manifest_rel,
            expected_train_sha256=train_sha, expected_train_rows=len(identity),
            expected_cycle_bounds=(47, 79),
        )
        for origin in ORIGINS:
            expected = labels[arrays[f"E{origin}_rows"].astype(np.int64)]
            if not np.array_equal(query_labels[origin], expected):
                raise AssertionError(f"Synthetic post-manifest label join mismatch for E{origin}")

        return {
            "status": "PASS",
            "adapter_binding": binding,
            "synthetic_train_rows": int(len(identity)),
            "query_rows_by_origin": {str(origin): int(len(arrays[f"E{origin}_rows"])) for origin in ORIGINS},
            "prediction_arrays": len(arrays),
            "prediction_archive_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "prediction_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "query_labels_blocked_until_manifest_commit": True,
            "query_labels_joined_after_commit": True,
            "missing_pos_fallback_bit_exact": all(
                np.array_equal(arrays[f"E{origin}_pPOS_UNIFORM"][~arrays[f"E{origin}_present"]],
                               arrays[f"E{origin}_pMISSING_HGB"][~arrays[f"E{origin}_present"]])
                for origin in ORIGINS),
            "primary_missing_reference_equals_pA_bit_exact": all(
                np.array_equal(arrays[f"E{origin}_MIX50_UNIFORM_PRIMARY"][~arrays[f"E{origin}_present"]],
                               arrays[f"E{origin}_pA"][~arrays[f"E{origin}_present"]])
                for origin in ORIGINS),
            "test_or_samplesubmission_accessed": False,
        }


class TestCLSPosOneSidedEvalAdapter(unittest.TestCase):
    def setUp(self) -> None:
        if hasattr(EVAL, "ARMS"):
            delattr(EVAL, "ARMS")

    def tearDown(self) -> None:
        if hasattr(EVAL, "ARMS"):
            delattr(EVAL, "ARMS")

    def test_exact_frozen_arm_binding(self) -> None:
        self.assertFalse(hasattr(EVAL, "ARMS"))
        binding = ADAPTER.install_frozen_arm_binding()
        self.assertEqual(binding["arms"], ["ONE_SIDED", "GLOBAL_STRENGTH"])
        self.assertFalse(binding["evaluator_source_modified"])

    def test_conflicting_arm_binding_is_rejected(self) -> None:
        EVAL.ARMS = ("UNIFORM",)
        with self.assertRaises(ADAPTER.AdapterError):
            ADAPTER.install_frozen_arm_binding()

    def test_full_synthetic_prepare_and_manifest_label_boundary(self) -> None:
        result = run_synthetic_prepare_entry_qa()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["query_labels_blocked_until_manifest_commit"])
        self.assertTrue(result["query_labels_joined_after_commit"])
        self.assertEqual(result["prediction_arrays"], 26)


if __name__ == "__main__":
    unittest.main()
