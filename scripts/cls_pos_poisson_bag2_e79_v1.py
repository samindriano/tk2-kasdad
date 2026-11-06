"""Source-gated E79 Poisson BAG2 runner.

The official path is intentionally split into three commands: freeze after
source commit, reserve/fit after one committed root pass, and one application
after that same pass plus two independently verified fit receipts.  Synthetic
CLI QA is isolated and never loads an official repository or file.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits


AUDIT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ID = "CLS_POS_POISSON_BAG2_E79_V1"
REPORT_REL = "reports/simplification_v1/CLS_POS_POISSON_BAG2_E79_V1"
PROTOCOL_REL = f"{REPORT_REL}/PROTOCOL.md"
SOURCE_READY_REL = f"{REPORT_REL}/SOURCE_READY.md"
FREEZE_REL = f"{REPORT_REL}/PREFIT_FREEZE.json"
ROOT_PASS_REL = f"{REPORT_REL}/ROOT_PASS.json"
HELPER_REL = "src/models/cls_pos_poisson_bag2_e79_v1.py"
RUNNER_REL = "scripts/cls_pos_poisson_bag2_e79_v1.py"
VERIFIER_REL = "scripts/verify_cls_pos_poisson_bag2_e79_v1.py"
TEST_REL = "tests/test_cls_pos_poisson_bag2_e79_v1.py"
SOURCE_FILES = (PROTOCOL_REL, SOURCE_READY_REL, HELPER_REL, RUNNER_REL, VERIFIER_REL, TEST_REL)

LEGACY_FREEZE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/GAMMA_SOURCE_FREEZE.json"
LEGACY_GAMMA_RUNNER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_gamma_v1.py"
LEGACY_INPUT_MANIFEST_REL = "reports/optimization_v3/posoffset_a02_endgame_v1/E79_SELECTED_MANIFEST.json"
LEGACY_SOURCE_RECIPE_REL = "reports/optimization_v3/a02_cgeom_benchmark_v1/FROZEN_RECIPES.json"
ORIGINAL_GAMMA_REL = (
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/units/"
    "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1.joblib"
)
ORIGINAL_GAMMA_RECEIPT_REL = ORIGINAL_GAMMA_REL[:-7] + ".json"
TRAIN_REL = "data/raw/classification/Train.csv"
TEST_REL = "data/raw/classification/Test.csv"
BASELINE_CSV_REL = (
    "submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/"
    "classification_a02_missing_temporal_e79.csv"
)
A02_TEST_ROOT_REL = "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed"
A02_TEST_NPZ_REL = f"{A02_TEST_ROOT_REL}/test_predictions.npz"
A02_TEST_RECEIPT_REL = f"{A02_TEST_ROOT_REL}/test_predictions.json"
A02_FIT_RECEIPT_REL = f"{A02_TEST_ROOT_REL}/fit_receipt.json"
A02_FIT_STATE_REL = f"{A02_TEST_ROOT_REL}/fit_state.json"
A02_MODEL_REL = f"{A02_TEST_ROOT_REL}/model.joblib"
A02_MANIFEST_REL = "reports/optimization_v3/a02_continuation_v1/submission_manifest.json"

OUTPUT_REL = f"experiments/{CANDIDATE_ID}"
RESERVATION_REL = f"{OUTPUT_REL}/RESERVATION.json"
LEDGER_REL = f"{OUTPUT_REL}/fit_ledger.jsonl"
FIT_QA_REL = f"{OUTPUT_REL}/FIT_QA.json"
APP_ATTEMPT_REL = f"{OUTPUT_REL}/APPLICATION_ATTEMPT.json"
APP_BUNDLE_REL = f"{OUTPUT_REL}/application_predictions.npz"
APP_RECEIPT_REL = f"{OUTPUT_REL}/APPLICATION_RECEIPT.json"
CSV_REL = "submissions/optimization_v3/cls_pos_poisson_bag2_e79_v1/classification_pos_poisson_bag2_e79.csv"

EXPECTED_TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
EXPECTED_A02_TEST_NPZ_SHA256 = "c267a4b998aca64982b6f63b10bedb2f3b3061007166db7731d221788ac89e47"
EXPECTED_A02_TEST_RECEIPT_SHA256 = "8aae0a6e2f16b707da53b16b0aef8f7395dc3a1d76fc1c7609edaed2304e84bc"
EXPECTED_A02_MODEL_SHA256 = "2791eddf410c7e81b119bde2a0b1d33a694eed4de84027c61cee97f6668dcb6e"
EXPECTED_A02_FIT_RECEIPT_SHA256 = "11a7e46b6595e92a421e358b8e7e3a5ffbac8a6db69c64c9c33fd3d5c5bd7d77"
EXPECTED_A02_MANIFEST_SHA256 = "3426e08002fd1eaa4f7caad64c210520932131d73224a72f7a56007b78cff9e3"
EXPECTED_BASELINE_CSV_SHA256 = "189a779b6100333a2fecfe443689fe7d340a331e828167221c81281a7ece4a4a"
EXPECTED_BASELINE_REF = 58001282
EXPECTED_BASELINE_PUBLIC_SCORE = "0.83225"
GIB = 1024 ** 3
MIN_FIT_START_BYTES = 2 * GIB
POST_FEATURE_RESERVE_BYTES = 1 * GIB
POST_FIT_RESERVE_BYTES = 512 * 1024 ** 2
MAX_THREADS = 2


def _load_helper(root: Path = AUDIT_ROOT) -> Any:
    path = root / HELPER_REL
    name = "_cls_pos_poisson_bag2_e79_helper"
    current = sys.modules.get(name)
    if current is not None:
        return current
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load E79 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load_helper()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_file_sha256(path: Path, relative: str) -> str:
    payload = path.read_bytes()
    if Path(relative).suffix.lower() in {".md", ".json"}:
        payload = payload.replace(b"\r\n", b"\n")
    return sha256_bytes(payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    H.require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout.decode("utf-8", errors="strict").strip()


def _git_succeeds(root: Path, *args: str) -> bool:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def _tree_bytes(root: Path, commit: str, relative: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{relative}"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"Committed source unavailable: {commit}:{relative}")
    return result.stdout


def _tree_stable_sha256(root: Path, commit: str, relative: str) -> str:
    body = _tree_bytes(root, commit, relative)
    if Path(relative).suffix.lower() in {".md", ".json"}:
        body = body.replace(b"\r\n", b"\n")
    return sha256_bytes(body)


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"Refusing to overwrite one-shot artifact: {path}") from exc
    return sha256_file(path)


def _write_json_once(path: Path, body: dict[str, Any]) -> str:
    payload = dict(body)
    payload["receipt_sha256"] = H.canonical_sha256(body)
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    file_sha = _write_once(path, raw)
    readback = _read_json(path)
    readback_sha = readback.pop("receipt_sha256", None)
    H.require(readback_sha == H.canonical_sha256(readback) and readback == body,
              f"JSON receipt readback mismatch: {path}")
    return file_sha


def _save_npz_once(path: Path, arrays: Mapping[str, Any]) -> str:
    for key, value in arrays.items():
        H.require(np.asarray(value).dtype.kind != "O", f"Object array not permitted in NPZ member {key}")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **dict(arrays))
    return _write_once(path, buffer.getvalue())


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_oldgamma(repository: Path) -> Any:
    """Import the frozen old gamma package from the explicit official checkout."""
    repo = repository.resolve()
    module_path = repo / LEGACY_GAMMA_RUNNER_REL
    H.require(module_path.is_file(), f"Pinned old gamma builder is missing: {module_path}")
    # The legacy builder uses absolute `scripts.*`/`src.*` imports.  Run this
    # official-only adapter in a clean CLI process and bind every imported
    # source file in PREFIT_FREEZE.json; no audit-worktree fallback is allowed.
    for name, module in list(sys.modules.items()):
        if name == "scripts" or name.startswith("scripts.") or name == "src" or name.startswith("src."):
            module_file = getattr(module, "__file__", None)
            if module_file is None or Path(module_file).resolve().is_relative_to(AUDIT_ROOT.resolve()):
                sys.modules.pop(name, None)
    repo_text = str(repo)
    sys.path[:] = [repo_text] + [item for item in sys.path if item != repo_text]
    module = importlib.import_module("scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1")
    H.require(Path(module.__file__).resolve() == module_path.resolve(),
              "Old gamma builder imported from a non-official repository")
    return module


def _resource_snapshot(stage: str) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    process = psutil.Process()
    info = process.memory_info()
    peak = getattr(info, "peak_wset", getattr(info, "peak_rss", None))
    return {
        "stage": stage,
        "available_bytes": int(vm.available),
        "rss_bytes": int(info.rss),
        "peak_rss_bytes": int(peak) if peak is not None else None,
        "max_threads": MAX_THREADS,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _require_resource(snapshot: Mapping[str, Any], minimum: int, label: str) -> None:
    H.require(int(snapshot["available_bytes"]) >= minimum,
              f"{label} RAM gate failed: available={snapshot['available_bytes']}, required={minimum}")


def _legacy_freeze_body(repository: Path) -> tuple[dict[str, Any], str]:
    path = repository / LEGACY_FREEZE_REL
    value = _read_json(path)
    body = dict(value)
    signature = body.pop("freeze_sha256", None)
    H.require(signature == H.canonical_sha256(body), "Legacy gamma source-freeze self-hash is invalid")
    H.require(value.get("status") == "FROZEN_SOURCE_PACKAGE_FOR_ONE_FULLTRAIN_GAMMA_FIT"
              and value.get("source_preparation_did_not_fit_gamma_or_read_test") is True,
              "Legacy gamma freeze has an unexpected status or boundary")
    return value, sha256_file(path)


def _verify_legacy_inputs(repository: Path, legacy_freeze: Mapping[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for rel, expected in sorted(legacy_freeze["input_file_sha256"].items()):
        path = repository / rel
        actual = stable_file_sha256(path, rel)
        H.require(actual == expected, f"Legacy immutable input hash changed: {rel}")
        resolved[rel] = actual
    source_hashes: dict[str, str] = {}
    for rel, expected in sorted(legacy_freeze["source_file_sha256"].items()):
        path = repository / rel
        actual = stable_file_sha256(path, rel)
        H.require(actual == expected, f"Legacy feature-builder source hash changed: {rel}")
        source_hashes[rel] = actual
    return {**{f"input::{k}": v for k, v in resolved.items()},
            **{f"source::{k}": v for k, v in source_hashes.items()}}


def build_source_freeze(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    repo = repository.resolve()
    source_commit = _git(audit_root, "rev-parse", "HEAD")
    status = _git(audit_root, "status", "--porcelain")
    H.require(not status, f"Audit worktree must be clean before freeze creation: {status}")
    source_hashes = {rel: stable_file_sha256(audit_root / rel, rel) for rel in SOURCE_FILES}
    for rel, actual in source_hashes.items():
        H.require(_tree_stable_sha256(audit_root, source_commit, rel) == actual,
                  f"Source is not committed at the freeze base: {rel}")

    official_head = _git(repo, "rev-parse", "HEAD")
    legacy, legacy_file_sha = _legacy_freeze_body(repo)
    legacy_inputs = _verify_legacy_inputs(repo, legacy)
    original_path = repo / ORIGINAL_GAMMA_REL
    original_receipt_path = repo / ORIGINAL_GAMMA_RECEIPT_REL
    H.require(sha256_file(original_path) == H.ORIGINAL_GAMMA_FILE_SHA256,
              "Pinned original full-champion gamma artifact differs from CGeom")
    H.require(original_receipt_path.is_file(), "Pinned original gamma receipt is missing")
    original_payload = joblib.load(original_path)
    original_gamma = H.extract_original_gamma(original_payload)
    H.require(H.sha256_array(original_gamma) == H.ORIGINAL_GAMMA_ARRAY_SHA256,
              "Pinned original gamma logical hash differs from CGeom attestation")

    recipes_path = repo / LEGACY_SOURCE_RECIPE_REL
    recipes = _read_json(recipes_path)
    pos_recipe = recipes.get("candidates", {}).get("POS_BAG_OFFSET", {})
    H.require("C=1" in pos_recipe.get("optimizer", "")
              and "C=1" in pos_recipe.get("objective", ""),
              "CGeom frozen recipe does not attest uniform C=1 gamma penalty")

    manifest_path = audit_root / LEGACY_INPUT_MANIFEST_REL
    input_manifest = _read_json(manifest_path)
    source = input_manifest.get("source_hashes", {})
    expected_external = {
        "official_test": EXPECTED_TEST_SHA256,
        "original_a02_test_bundle": EXPECTED_A02_TEST_NPZ_SHA256,
        "original_a02_test_receipt": EXPECTED_A02_TEST_RECEIPT_SHA256,
        "original_a02_model": EXPECTED_A02_MODEL_SHA256,
        "original_a02_fit_receipt": EXPECTED_A02_FIT_RECEIPT_SHA256,
        "original_a02_manifest": EXPECTED_A02_MANIFEST_SHA256,
    }
    for key, expected in expected_external.items():
        H.require(source.get(key) == expected,
                  f"Frozen A02/Test metadata differs from expected lineage: {key}")
    H.require(input_manifest.get("rows") == 29_421
              and input_manifest.get("cycle_range") == [80, 91]
              and input_manifest.get("test_targets_loaded") is False,
              "Historical E79 application manifest has an unexpected data boundary")

    body: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "FROZEN_SOURCE_ONLY_WAITING_ROOT_PASS",
        "audit_root": str(audit_root.resolve()),
        "audit_branch": _git(audit_root, "branch", "--show-current"),
        "audit_source_commit": source_commit,
        "audit_source_hashes": source_hashes,
        "official_repository": str(repo),
        "official_repository_head": official_head,
        "legacy_gamma_source_freeze_file_sha256": legacy_file_sha,
        "legacy_gamma_source_freeze_signature": legacy["freeze_sha256"],
        "legacy_source_and_input_hashes": legacy_inputs,
        "official_train": {
            "path": TRAIN_REL,
            "rows": H.TRAIN_ROWS,
            "sha256": H.OFFICIAL_TRAIN_SHA256,
            "cycles": [47, 79],
            "fit_target_read_during_source_prep": False,
        },
        "full_champion_gamma": {
            "path": ORIGINAL_GAMMA_REL,
            "artifact_sha256": H.ORIGINAL_GAMMA_FILE_SHA256,
            "receipt_path": ORIGINAL_GAMMA_RECEIPT_REL,
            "receipt_sha256": sha256_file(original_receipt_path),
            "payload_member": "gamma",
            "gamma_sha256": H.sha256_array(original_gamma),
            "shape": list(original_gamma.shape),
            "fit_rows": H.TRAIN_ROWS,
            "recipe": "CGeom E79 POS_BAG_OFFSET; uniform ridge C=1",
        },
        "feature_builder": {
            "module_path": LEGACY_GAMMA_RUNNER_REL,
            "function": "g._build_train_features",
            "application_function": "g._build_feature_matrices_for_rows",
            "exact_base_matrix_shape": list(H.EXPECTED_BASE_SHAPE),
            "exact_base_matrix_nnz": H.EXPECTED_BASE_NNZ,
            "exact_base_matrix_sha256": H.EXPECTED_BASE_SHA256,
            "exact_position_matrix_shape": list(H.EXPECTED_POSITION_SHAPE),
            "exact_position_matrix_nnz": H.EXPECTED_POSITION_NNZ,
            "exact_position_matrix_sha256": H.EXPECTED_POSITION_SHA256,
            "exact_ordered_train_ids_sha256": H.EXPECTED_ORDERED_TRAIN_IDS_SHA256,
            "global_strength_penalty_or_scalar_used": False,
            "target_load_order": "feature/matrix parity first; artifact_signal only immediately before the two authorized fits",
        },
        "fixed_recipe": {
            "fit_cap": 2,
            "fit_ids": list(H.FIT_IDS),
            "seeds": list(H.SEEDS),
            "poisson_lambda": 1.0,
            "weight_normalization": "sum(weights)=50608",
            "ridge_c": H.RIDGE_C,
            "objective": "sum_i a_i*logaddexp(0,-(2*y_i-1)*(z0_i+Xpos_i@gamma)) + 0.5*||gamma||_2^2",
            "initial_gamma": "zero",
            "optimizer": "scipy.optimize.minimize:L-BFGS-B",
            "optimizer_options": H.OPTIMIZER_OPTIONS,
            "new_preprocessing_fits": 0,
            "new_original_head_fits": 0,
        },
        "fixed_application": {
            "present_probability": "0.5*mean(p_original,p17,p137) + 0.5*pA02",
            "missing_probability": "pA02 exactly",
            "present_threshold": H.PRESENT_THRESHOLD,
            "missing_threshold": H.MISSING_THRESHOLD,
            "threshold_operator": ">=",
            "baseline_csv_path": BASELINE_CSV_REL,
            "baseline_csv_sha256": EXPECTED_BASELINE_CSV_SHA256,
            "baseline_public_ref": EXPECTED_BASELINE_REF,
            "baseline_public_score": EXPECTED_BASELINE_PUBLIC_SCORE,
            "baseline_used_for_selection": False,
            "baseline_decision_replay_required_before_candidate_csv": True,
        },
        "application_input_lineage": {
            "official_test_sha256_expected_from_frozen_prior_receipt": EXPECTED_TEST_SHA256,
            "a02_test_npz_sha256_expected": EXPECTED_A02_TEST_NPZ_SHA256,
            "a02_test_receipt_sha256_expected": EXPECTED_A02_TEST_RECEIPT_SHA256,
            "a02_model_sha256_expected": EXPECTED_A02_MODEL_SHA256,
            "a02_fit_receipt_sha256_expected": EXPECTED_A02_FIT_RECEIPT_SHA256,
            "a02_manifest_sha256_expected": EXPECTED_A02_MANIFEST_SHA256,
            "source_of_expected_hashes": str(LEGACY_INPUT_MANIFEST_REL),
            "test_or_sample_opened_during_source_prep": False,
        },
        "resource_policy": {
            "max_threads": MAX_THREADS,
            "fit_start_minimum_available_bytes": MIN_FIT_START_BYTES,
            "post_feature_reserve_bytes": POST_FEATURE_RESERVE_BYTES,
            "post_fit_reserve_bytes": POST_FIT_RESERVE_BYTES,
            "one_shared_heavy_lock": True,
            "fit_retry_allowed": False,
        },
        "data_boundary": {
            "train_target_loaded_during_source_prep": False,
            "test_raw_opened_during_source_prep": False,
            "test_prediction_values_opened_during_source_prep": False,
            "sample_submission_opened": False,
            "test_labels_accessed": False,
            "kaggle_contacted": False,
        },
    }
    body["freeze_signature"] = H.canonical_sha256(body)
    return body


def create_source_freeze(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    path = audit_root / FREEZE_REL
    body = build_source_freeze(repository, audit_root=audit_root)
    _write_json_once(path, {k: v for k, v in body.items() if k != "freeze_signature"} |
                     {"freeze_signature": body["freeze_signature"]})
    return {"status": body["status"], "freeze_signature": body["freeze_signature"],
            "freeze_file_sha256": sha256_file(path), "path": FREEZE_REL}


def verify_source_freeze(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    path = audit_root / FREEZE_REL
    freeze = _read_json(path)
    receipt_sha = freeze.get("receipt_sha256")
    receipt_body = dict(freeze)
    receipt_body.pop("receipt_sha256", None)
    H.require(receipt_sha == H.canonical_sha256(receipt_body),
              "E79 prefit freeze receipt self-hash is invalid")
    signature = receipt_body.get("freeze_signature")
    body = dict(receipt_body)
    body.pop("freeze_signature", None)
    H.require(signature == H.canonical_sha256(body), "E79 prefit freeze signature is invalid")
    H.require(freeze.get("candidate_id") == CANDIDATE_ID
              and freeze.get("status") == "FROZEN_SOURCE_ONLY_WAITING_ROOT_PASS",
              "E79 prefit freeze status/candidate mismatch")
    repo = repository.resolve()
    H.require(str(repo) == freeze.get("official_repository"), "Official repository root changed")
    H.require(_git(repo, "rev-parse", "HEAD") == freeze.get("official_repository_head"),
              "Official repository HEAD changed after E79 source freeze")
    H.require(_git_succeeds(audit_root, "merge-base", "--is-ancestor",
                            freeze["audit_source_commit"], "HEAD"),
              "Audit source commit is not an ancestor of the current HEAD")
    for rel, expected in freeze["audit_source_hashes"].items():
        H.require(stable_file_sha256(audit_root / rel, rel) == expected,
                  f"Committed E79 source changed: {rel}")
        H.require(_tree_stable_sha256(audit_root, freeze["audit_source_commit"], rel) == expected,
                  f"E79 source commit does not bind source file: {rel}")
    legacy, legacy_sha = _legacy_freeze_body(repo)
    H.require(legacy_sha == freeze["legacy_gamma_source_freeze_file_sha256"]
              and legacy["freeze_sha256"] == freeze["legacy_gamma_source_freeze_signature"],
              "Legacy full-Train feature source freeze changed")
    current_legacy_inputs = _verify_legacy_inputs(repo, legacy)
    H.require(current_legacy_inputs == freeze["legacy_source_and_input_hashes"],
              "Legacy feature-builder source/input inventory changed")
    H.require(sha256_file(repo / ORIGINAL_GAMMA_REL) == H.ORIGINAL_GAMMA_FILE_SHA256
              and sha256_file(repo / ORIGINAL_GAMMA_RECEIPT_REL)
              == freeze["full_champion_gamma"]["receipt_sha256"],
              "Original full-champion gamma/receipt changed")
    H.require(sha256_file(repo / TRAIN_REL) == H.OFFICIAL_TRAIN_SHA256,
              "Official Train hash changed")
    H.require(_git(repo, "rev-parse", "HEAD") == freeze["official_repository_head"],
              "Official repository changed while verifying E79 freeze")
    return {"status": "PASS_E79_SOURCE_FREEZE", "freeze_signature": signature,
            "freeze_file_sha256": sha256_file(path), "audit_source_commit": freeze["audit_source_commit"],
            "official_repository_head": freeze["official_repository_head"]}


def verify_root_pass(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    freeze_check = verify_source_freeze(repository, audit_root=audit_root)
    path = audit_root / ROOT_PASS_REL
    H.require(path.is_file(), "Committed Sol root pass is required before reserve, fit, or Test application")
    passed = _read_json(path)
    body = dict(passed)
    pass_sha = body.pop("pass_sha256", None)
    H.require(pass_sha == H.canonical_sha256(body), "Root pass self-hash is invalid")
    head = _git(audit_root, "rev-parse", "HEAD")
    H.require(passed.get("reviewed_head") == head
              and _tree_stable_sha256(audit_root, head, ROOT_PASS_REL) == stable_file_sha256(path, ROOT_PASS_REL),
              "Root pass is not committed at its reviewed HEAD")
    H.require(passed.get("status") == "PASS_CLS_POS_POISSON_BAG2_E79_SOURCE_AND_APPLICATION"
              and passed.get("candidate_id") == CANDIDATE_ID
              and passed.get("prefit_freeze_signature") == freeze_check["freeze_signature"]
              and passed.get("prefit_freeze_file_sha256") == freeze_check["freeze_file_sha256"],
              "Root pass does not approve this exact E79 source freeze")
    H.require(passed.get("authorized_fit_ids") == list(H.FIT_IDS)
              and passed.get("fit_cap") == 2
              and passed.get("retry_allowed") is False,
              "Root pass does not authorize exactly the two fixed fits")
    H.require(passed.get("test_application_authorized") is True
              and passed.get("one_test_application_only") is True
              and passed.get("baseline_decision_replay_required") is True
              and passed.get("test_target_access_authorized") is False
              and passed.get("sample_submission_access_authorized") is False
              and passed.get("kaggle_access_authorized") is False,
              "Root pass application/data boundaries are incomplete")
    freeze = _read_json(audit_root / FREEZE_REL)
    for key, expected in (
        ("official_test_sha256", EXPECTED_TEST_SHA256),
        ("a02_test_npz_sha256", EXPECTED_A02_TEST_NPZ_SHA256),
        ("a02_test_receipt_sha256", EXPECTED_A02_TEST_RECEIPT_SHA256),
        ("a02_model_sha256", EXPECTED_A02_MODEL_SHA256),
        ("a02_fit_receipt_sha256", EXPECTED_A02_FIT_RECEIPT_SHA256),
        ("a02_manifest_sha256", EXPECTED_A02_MANIFEST_SHA256),
        ("baseline_csv_sha256", EXPECTED_BASELINE_CSV_SHA256),
    ):
        H.require(passed.get(key) == expected,
                  f"Root pass does not bind frozen E79 application input: {key}")
    return {"status": "PASS_E79_ROOT_PASS", "pass_sha256": pass_sha,
            "pass_file_sha256": sha256_file(path), "reviewed_head": head,
            "freeze_signature": freeze["freeze_signature"]}


def _ledger_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            H.require(isinstance(value, dict), "Ledger rows must be JSON objects")
            result.append(value)
    return result


def reserve_fits(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    freeze_check = verify_source_freeze(repository, audit_root=audit_root)
    root_pass = verify_root_pass(repository, audit_root=audit_root)
    output = audit_root / OUTPUT_REL
    reservation_path = audit_root / RESERVATION_REL
    ledger_path = audit_root / LEDGER_REL
    expected_body = {
        "schema_version": 1,
        "candidate_id": CANDIDATE_ID,
        "status": "EXACT_TWO_E79_FITS_RESERVED",
        "fit_cap": 2,
        "fit_ids": list(H.FIT_IDS),
        "freeze_signature": freeze_check["freeze_signature"],
        "root_pass_sha256": root_pass["pass_sha256"],
        "retry_allowed": False,
        "preprocessing_fits": 0,
        "original_head_fits": 0,
    }
    if reservation_path.exists():
        saved = _read_json(reservation_path)
        body = dict(saved)
        receipt_sha = body.pop("receipt_sha256", None)
        H.require(receipt_sha == H.canonical_sha256(body) and body == expected_body,
                  "Existing E79 reservation differs; no reset/overwrite is allowed")
        return {"status": expected_body["status"], "fit_ids": list(H.FIT_IDS),
                "reservation_file_sha256": sha256_file(reservation_path), "resumed": True}
    H.require(not output.exists(), "E79 output namespace already exists without its reservation")
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_once(reservation_path, expected_body)
    _append_event(ledger_path, {"event": "FIT_CAP_RESERVED", **expected_body})
    return {"status": expected_body["status"], "fit_ids": list(H.FIT_IDS),
            "reservation_file_sha256": sha256_file(reservation_path), "resumed": False}


def _fit_paths(audit_root: Path, fit_id: str) -> tuple[Path, Path, Path]:
    base = audit_root / OUTPUT_REL / "fits" / fit_id
    return base / "ATTEMPT.json", base / "gamma.npz", base / "FIT_RECEIPT.json"


def _load_fit_gamma(audit_root: Path, fit_id: str, freeze_signature: str) -> tuple[np.ndarray, dict[str, Any]]:
    verifier = _load_verifier(audit_root)
    return verifier.verify_fit_artifact(audit_root / OUTPUT_REL, fit_id,
                                        freeze_signature=freeze_signature)


def _load_verifier(audit_root: Path = AUDIT_ROOT) -> Any:
    path = audit_root / VERIFIER_REL
    name = "_cls_pos_poisson_bag2_e79_independent_verifier"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    H.require(spec is not None and spec.loader is not None, "Unable to load independent E79 verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_poisson_weights(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return H.load_pilot_helpers().poisson_row_weights(n_rows, seed)


def fit_all(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    repo = repository.resolve()
    freeze_check = verify_source_freeze(repo, audit_root=audit_root)
    root_pass = verify_root_pass(repo, audit_root=audit_root)
    reservation_path = audit_root / RESERVATION_REL
    H.require(reservation_path.is_file(), "Run reserve after root pass before any official fit")
    reservation = _read_json(reservation_path)
    H.require(reservation.get("status") == "EXACT_TWO_E79_FITS_RESERVED"
              and reservation.get("fit_ids") == list(H.FIT_IDS)
              and reservation.get("root_pass_sha256") == root_pass["pass_sha256"],
              "Reservation does not match current root pass and frozen fit IDs")
    output = audit_root / OUTPUT_REL
    ledger_path = audit_root / LEDGER_REL
    verifier = _load_verifier(audit_root)
    existing: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    pending: list[tuple[int, str]] = []
    events = _ledger_events(ledger_path)
    for seed in H.SEEDS:
        fit_id = H.fit_id_for_seed(seed)
        attempt, gamma_path, receipt_path = _fit_paths(audit_root, fit_id)
        if receipt_path.exists() or gamma_path.exists():
            H.require(attempt.exists() and gamma_path.exists() and receipt_path.exists(),
                      f"Partial output exists for {fit_id}; no retry or overwrite")
            existing[fit_id] = _load_fit_gamma(audit_root, fit_id, freeze_check["freeze_signature"])
        elif attempt.exists() or any(e.get("fit_id") == fit_id and e.get("event") == "FIT_STARTED" for e in events):
            raise RuntimeError(f"Fit ID already consumed without a saved receipt: {fit_id}; no retry")
        else:
            pending.append((seed, fit_id))
    if not pending:
        qa = verifier.verify_fit_set(audit_root / OUTPUT_REL, freeze_signature=freeze_check["freeze_signature"])
        return {"status": "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED", **qa, "resumed": True}

    gamma_runner = _load_oldgamma(repo)
    with threadpool_limits(limits=MAX_THREADS), ExitStack() as stack:
        locks = [stack.enter_context(gamma_runner.r3._shared_job_lock()),
                 stack.enter_context(gamma_runner.r3._cgeom_local_lock(repo))]
        resource_before = _resource_snapshot("before exact-parity full-Train feature reconstruction")
        _require_resource(resource_before, MIN_FIT_START_BYTES, "E79 fit start")
        built = gamma_runner._build_train_features(repo)
        base, position, offset = H.validate_matrix_bundle(built)
        resource_features = _resource_snapshot("after exact-parity full-Train feature reconstruction")
        _require_resource(resource_features, POST_FEATURE_RESERVE_BYTES, "E79 post-feature reserve")
        # The old builder exposes metadata plus the exact feature arrays.  This
        # sole target-only read occurs only after it has checked matrix/ID parity.
        target = gamma_runner._load_train_target_last(repo)
        H.require(target.shape == (H.TRAIN_ROWS,) and np.isin(target, (0, 1)).all(),
                  "Official artifact_signal vector does not match frozen full-Train population")
        target_sha = H.sha256_array(target)
        original_path = repo / ORIGINAL_GAMMA_REL
        H.require(sha256_file(original_path) == H.ORIGINAL_GAMMA_FILE_SHA256,
                  "Original gamma artifact changed after feature parity")
        payload = joblib.load(original_path)
        gamma_original = H.extract_original_gamma(payload, expected_length=position.shape[1])
        H.require(H.sha256_array(gamma_original) == H.ORIGINAL_GAMMA_ARRAY_SHA256,
                  "Original full-champion gamma logical hash changed")
        base_hash = built["matrices"]["base"]["logical_sha256"]
        position_hash = built["matrices"]["position"]["logical_sha256"]
        input_hashes = {
            "train_file_sha256": H.OFFICIAL_TRAIN_SHA256,
            "ordered_train_ids_sha256": built["ids_sha256"],
            "base_matrix_sha256": base_hash,
            "position_matrix_sha256": position_hash,
            "c2b_offset_sha256": H.sha256_array(offset),
            "artifact_signal_sha256": target_sha,
            "original_gamma_artifact_sha256": H.ORIGINAL_GAMMA_FILE_SHA256,
            "original_gamma_array_sha256": H.sha256_array(gamma_original),
        }
        for seed, fit_id in pending:
            attempt_path, gamma_path, receipt_path = _fit_paths(audit_root, fit_id)
            H.require(not attempt_path.exists() and not gamma_path.exists() and not receipt_path.exists(),
                      f"One-shot output appeared before fit start: {fit_id}")
            raw_counts, weights = _load_poisson_weights(H.TRAIN_ROWS, seed)
            H.require(int(raw_counts.sum()) > 0 and np.isclose(weights.sum(), H.TRAIN_ROWS,
                      rtol=0.0, atol=1e-10 * H.TRAIN_ROWS),
                      f"Frozen Poisson weights invalid for {fit_id}")
            resource_fit = _resource_snapshot(f"immediately before {fit_id}")
            _require_resource(resource_fit, POST_FEATURE_RESERVE_BYTES, f"{fit_id} fit reserve")
            start_event = {
                "event": "FIT_STARTED", "candidate_id": CANDIDATE_ID, "fit_id": fit_id,
                "seed": seed, "fit_rows": H.TRAIN_ROWS, "fit_cap": 2,
                "freeze_signature": freeze_check["freeze_signature"],
                "root_pass_sha256": root_pass["pass_sha256"],
                "source_commit": freeze_check["audit_source_commit"],
                "input_hashes": input_hashes,
                "poisson_raw_counts_sha256": H.sha256_array(raw_counts),
                "normalized_weights_sha256": H.sha256_array(weights),
                "normalized_weights_sum": float(weights.sum()),
                "resource_before": resource_fit,
                "locks": [{"mechanism": "OS nonblocking advisory lock", "acquired": True,
                           "path": "shared+official-CGeom"}],
                "one_attempt_no_retry": True,
                "test_or_sample_access": False,
            }
            _write_once(attempt_path, (json.dumps(start_event, sort_keys=True,
                                                  ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode())
            _append_event(ledger_path, start_event)
            try:
                gamma, optimizer = H.load_pilot_helpers().fit_poisson_weighted_offset(
                    position, offset, target, weights)
            except Exception as exc:
                _append_event(ledger_path, {"event": "FIT_FAILED", "candidate_id": CANDIDATE_ID,
                                            "fit_id": fit_id, "seed": seed,
                                            "freeze_signature": freeze_check["freeze_signature"],
                                            "error_type": type(exc).__name__, "error": str(exc),
                                            "retry_allowed": False})
                raise RuntimeError(f"Consumed E79 fit failed; no retry: {fit_id}: {exc}") from exc
            resource_after = _resource_snapshot(f"after {fit_id}")
            _require_resource(resource_after, POST_FIT_RESERVE_BYTES, f"{fit_id} post-fit reserve")
            artifact_sha = _save_npz_once(gamma_path, {"gamma": gamma.astype(np.float64)})
            receipt_body = {
                "schema_version": 1,
                "candidate_id": CANDIDATE_ID,
                "fit_id": fit_id,
                "seed": seed,
                "status": "PASS_E79_POISSON_GAMMA_FIT_PERSISTED",
                "mode": "official",
                "source_commit": freeze_check["audit_source_commit"],
                "freeze_signature": freeze_check["freeze_signature"],
                "freeze_file_sha256": freeze_check["freeze_file_sha256"],
                "root_pass_sha256": root_pass["pass_sha256"],
                "root_pass_file_sha256": root_pass["pass_file_sha256"],
                "fit_population": {"rows": H.TRAIN_ROWS, "cycles": [47, 79],
                                   "source": "official Classification Train"},
                "input_hashes": input_hashes,
                "poisson_lambda": 1.0,
                "raw_counts_sha256": H.sha256_array(raw_counts),
                "normalized_weights_sha256": H.sha256_array(weights),
                "normalized_weights_sum": float(weights.sum()),
                "normalized_weights_zero_count": int(np.count_nonzero(weights == 0.0)),
                "objective": "sum_i a_i*logaddexp(0,-(2*y_i-1)*(z0_i+Xpos_i@gamma)) + 0.5*||gamma||_2^2",
                "ridge_c": 1.0,
                "global_strength_penalty_used": False,
                "optimizer": optimizer,
                "gamma_shape": list(gamma.shape),
                "gamma_dtype": str(gamma.dtype),
                "gamma_sha256": H.sha256_array(gamma),
                "artifact_path": str(gamma_path.relative_to(audit_root)).replace("\\", "/"),
                "artifact_sha256": artifact_sha,
                "attempt_path": str(attempt_path.relative_to(audit_root)).replace("\\", "/"),
                "resource_before": resource_fit,
                "resource_after": resource_after,
                "preprocessing_fit_count": 0,
                "original_head_fit_count": 0,
                "fit_count": 1,
                "retry_allowed": False,
                "test_or_sample_access": False,
            }
            receipt_file_sha = _write_json_once(receipt_path, receipt_body)
            fitted_gamma, verified_receipt = verifier.verify_fit_artifact(
                output, fit_id, freeze_signature=freeze_check["freeze_signature"])
            H.require(np.array_equal(fitted_gamma, gamma)
                      and verified_receipt["artifact_sha256"] == artifact_sha,
                      f"Independent fit readback differs for {fit_id}")
            _append_event(ledger_path, {"event": "FIT_SAVED", "candidate_id": CANDIDATE_ID,
                                        "fit_id": fit_id, "seed": seed,
                                        "artifact_sha256": artifact_sha,
                                        "receipt_file_sha256": receipt_file_sha,
                                        "freeze_signature": freeze_check["freeze_signature"],
                                        "root_pass_sha256": root_pass["pass_sha256"]})
            existing[fit_id] = (fitted_gamma, verified_receipt)
    qa = verifier.verify_fit_set(output, freeze_signature=freeze_check["freeze_signature"])
    qa_body = {"schema_version": 1, "candidate_id": CANDIDATE_ID,
               "status": "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED",
               "freeze_signature": freeze_check["freeze_signature"],
               "fit_ids": list(H.FIT_IDS), "fit_cap": 2,
               "fit_receipts": qa["fit_receipts"],
               "test_or_sample_access": False, "kaggle_contacted": False}
    qa_path = audit_root / FIT_QA_REL
    if qa_path.exists():
        H.require(_read_json(qa_path).get("receipt_sha256") == H.canonical_sha256(qa_body),
                  "Existing E79 FIT_QA differs; refusing overwrite")
    else:
        _write_json_once(qa_path, qa_body)
    return {"status": qa_body["status"], **qa, "resumed": bool(existing)}


def _read_baseline_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={"record_id": str}, keep_default_na=False)
    H.require(list(frame.columns) == ["record_id", "artifact_signal"],
              "Frozen champion baseline CSV schema differs")
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    raw_labels = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy()
    H.require(len(ids) == 29_421 and len(set(ids.tolist())) == len(ids),
              "Frozen champion baseline CSV row/ID count differs")
    H.require(raw_labels.dtype.kind in "biuf" and np.isfinite(raw_labels).all()
              and np.isin(raw_labels, [0, 1]).all(), "Frozen champion baseline has non-binary labels")
    return ids, raw_labels.astype(np.int8, copy=False)


def apply_test_once(repository: Path, *, audit_root: Path = AUDIT_ROOT) -> dict[str, Any]:
    repo = repository.resolve()
    freeze_check = verify_source_freeze(repo, audit_root=audit_root)
    root_pass = verify_root_pass(repo, audit_root=audit_root)
    output = audit_root / OUTPUT_REL
    verifier = _load_verifier(audit_root)
    fit_qa_path = audit_root / FIT_QA_REL
    H.require(fit_qa_path.is_file(), "Independent two-fit QA must pass before Test application")
    fit_qa = _read_json(fit_qa_path)
    H.require(fit_qa.get("status") == "PASS_TWO_E79_FITS_INDEPENDENTLY_VERIFIED"
              and fit_qa.get("fit_ids") == list(H.FIT_IDS)
              and fit_qa.get("freeze_signature") == freeze_check["freeze_signature"],
              "Test application requires the exact frozen two-fit QA")
    verifier.verify_fit_set(output, freeze_signature=freeze_check["freeze_signature"])
    attempt_path = audit_root / APP_ATTEMPT_REL
    bundle_path = audit_root / APP_BUNDLE_REL
    receipt_path = audit_root / APP_RECEIPT_REL
    csv_path = repo / CSV_REL
    if receipt_path.exists() or bundle_path.exists() or csv_path.exists():
        H.require(attempt_path.exists() and receipt_path.exists() and bundle_path.exists() and csv_path.exists(),
                  "Partial one-shot application output exists; no retry")
        return verifier.verify_application(output, receipt_path=receipt_path,
                                           bundle_path=bundle_path, csv_path=csv_path,
                                           baseline_csv_path=repo / BASELINE_CSV_REL,
                                           expected_baseline_sha256=EXPECTED_BASELINE_CSV_SHA256)
    H.require(not attempt_path.exists(), "An E79 Test application attempt is already consumed; no retry")

    gamma_runner = _load_oldgamma(repo)
    with threadpool_limits(limits=MAX_THREADS), ExitStack() as stack:
        locks = [stack.enter_context(gamma_runner.r3._shared_job_lock()),
                 stack.enter_context(gamma_runner.r3._cgeom_local_lock(repo))]
        resource_before = _resource_snapshot("before single authorized E79 Test application")
        _require_resource(resource_before, GIB, "E79 Test application")
        marker = {
            "schema_version": 1, "candidate_id": CANDIDATE_ID,
            "status": "ONE_E79_TEST_APPLICATION_ATTEMPT_STARTED",
            "source_commit": _git(audit_root, "rev-parse", "HEAD"),
            "freeze_signature": freeze_check["freeze_signature"],
            "root_pass_sha256": root_pass["pass_sha256"],
            "fit_receipt_files_sha256": {fit_id: sha256_file(_fit_paths(audit_root, fit_id)[2])
                                         for fit_id in H.FIT_IDS},
            "resource_before": resource_before,
            "locks": locks,
            "baseline_replay_required_before_candidate_csv": True,
            "test_target_requested_or_loaded": False,
            "sample_submission_loaded": False,
            "kaggle_contacted": False,
            "one_application_no_retry": True,
        }
        _write_once(attempt_path, (json.dumps(marker, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False, indent=2) + "\n").encode())

        test_path = repo / TEST_REL
        H.require(sha256_file(test_path) == EXPECTED_TEST_SHA256,
                  "Official Test raw feature file hash differs from frozen receipt")
        test, columns = gamma_runner.r3.read_feature_csv(
            test_path, gamma_runner.cgeom.CLASSIFICATION_FEATURES)
        H.require(len(test) == 29_421 and not (set(test.columns) & {"artifact_signal", "threat_rank"}),
                  "Feature-only Test population/schema differs")
        ids = test.record_id.astype(str).to_numpy(dtype=str)
        H.require(not pd.isna(ids).any() and len(set(ids.tolist())) == len(ids),
                  "Official Test record IDs are missing or duplicated")
        cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
        H.require(np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)),
                  "Official Test cycles differ from frozen 80-91 range")

        a02_npz_path = repo / A02_TEST_NPZ_REL
        a02_test_receipt_path = repo / A02_TEST_RECEIPT_REL
        a02_fit_receipt_path = repo / A02_FIT_RECEIPT_REL
        a02_state_path = repo / A02_FIT_STATE_REL
        a02_model_path = repo / A02_MODEL_REL
        a02_manifest_path = repo / A02_MANIFEST_REL
        bindings = {
            "official_test_sha256": sha256_file(test_path),
            "a02_test_npz_sha256": sha256_file(a02_npz_path),
            "a02_test_receipt_sha256": sha256_file(a02_test_receipt_path),
            "a02_fit_receipt_sha256": sha256_file(a02_fit_receipt_path),
            "a02_model_sha256": sha256_file(a02_model_path),
            "a02_manifest_sha256": sha256_file(a02_manifest_path),
            "baseline_csv_sha256": sha256_file(repo / BASELINE_CSV_REL),
        }
        expected_bindings = {
            "official_test_sha256": EXPECTED_TEST_SHA256,
            "a02_test_npz_sha256": EXPECTED_A02_TEST_NPZ_SHA256,
            "a02_test_receipt_sha256": EXPECTED_A02_TEST_RECEIPT_SHA256,
            "a02_fit_receipt_sha256": EXPECTED_A02_FIT_RECEIPT_SHA256,
            "a02_model_sha256": EXPECTED_A02_MODEL_SHA256,
            "a02_manifest_sha256": EXPECTED_A02_MANIFEST_SHA256,
            "baseline_csv_sha256": EXPECTED_BASELINE_CSV_SHA256,
        }
        for key, expected in expected_bindings.items():
            H.require(bindings[key] == expected and root_pass.get(key) == expected,
                      f"Application root pass/input binding mismatch: {key}")
        a02_test_receipt = _read_json(a02_test_receipt_path)
        a02_fit_receipt = _read_json(a02_fit_receipt_path)
        a02_manifest = _read_json(a02_manifest_path)
        a02_fit_state = _read_json(a02_state_path)
        H.require(a02_test_receipt.get("npz_sha256") == bindings["a02_test_npz_sha256"]
                  and a02_test_receipt.get("fit_receipt_sha256") == a02_fit_receipt.get("receipt_sha256")
                  and a02_test_receipt.get("freeze_signature") == a02_fit_receipt.get("freeze_signature")
                  and a02_test_receipt.get("official_test_targets_loaded") is False,
                  "A02 Test prediction receipt chain does not verify")
        H.require(a02_fit_receipt.get("model_sha256") == bindings["a02_model_sha256"]
                  and a02_fit_receipt.get("test_data_used") is False
                  and a02_fit_receipt.get("query_labels_passed_to_fit") is False
                  and a02_fit_receipt.get("fit_rows") in (None, H.TRAIN_ROWS)
                  and a02_manifest.get("test_target_loaded") is False
                  and a02_manifest.get("official_train_rows") == H.TRAIN_ROWS
                  and a02_fit_state == joblib.load(a02_model_path).audit_state(),
                  "A02 full-Train fitted-state/test-target lineage failed")
        with np.load(a02_npz_path, allow_pickle=False) as archive:
            required = {"record_ids", "candidate_probability"}
            H.require(required.issubset(set(archive.files)), "A02 pA cache lacks required prediction members")
            a02_ids = np.asarray(archive["record_ids"]).astype(str).reshape(-1)
            p_a02 = np.asarray(archive["candidate_probability"], dtype=np.float64).reshape(-1)
        H.require(np.array_equal(a02_ids, ids) and len(p_a02) == len(ids)
                  and np.isfinite(p_a02).all() and np.all((p_a02 >= 0.0) & (p_a02 <= 1.0)),
                  "A02 pA ordered IDs/probabilities do not align with official Test")

        built = gamma_runner._build_feature_matrices_for_rows(repo, test)
        position = sparse.csr_matrix(built["position"], dtype=np.float64)
        offset = np.asarray(built["offset"], dtype=np.float64).reshape(-1)
        missing = np.asarray(built["missing_mask"], dtype=bool).reshape(-1)
        H.require(position.shape == (len(ids), H.EXPECTED_POSITION_SHAPE[1])
                  and len(offset) == len(ids) and len(missing) == len(ids)
                  and np.isfinite(position.data).all() and np.isfinite(offset).all(),
                  "Target-free Test feature transformation failed schema/finiteness")
        present = ~missing
        original_payload = joblib.load(repo / ORIGINAL_GAMMA_REL)
        gamma_original = H.extract_original_gamma(original_payload, expected_length=position.shape[1])
        gamma17, rec17 = _load_fit_gamma(audit_root, H.fit_id_for_seed(17), freeze_check["freeze_signature"])
        gamma137, rec137 = _load_fit_gamma(audit_root, H.fit_id_for_seed(137), freeze_check["freeze_signature"])
        policies = H.build_probabilities(position, offset, gamma_original, gamma17, gamma137,
                                          p_a02, present)
        champion_labels = H.apply_thresholds(policies["probability_champion"], present)
        candidate_labels = H.apply_thresholds(policies["probability_candidate"], present)
        H.require(np.array_equal(policies["probability_champion"][missing], p_a02[missing])
                  and np.array_equal(policies["probability_candidate"][missing], p_a02[missing]),
                  "Missing-stream probability fallback is not exact A02 pA")
        baseline_path = repo / BASELINE_CSV_REL
        H.require(sha256_file(baseline_path) == EXPECTED_BASELINE_CSV_SHA256,
                  "Frozen public-champion reference CSV hash differs")
        baseline_ids, baseline_labels = _read_baseline_csv(baseline_path)
        H.require(np.array_equal(baseline_ids, ids), "Champion baseline ID order differs from official Test")
        H.require(np.array_equal(champion_labels, baseline_labels),
                  "Original uniform-gamma full-champion decisions fail exact baseline parity")

        # The candidate CSV is materialized only after full-champion decision
        # replay succeeds on every ordered Test ID.
        bundle_sha = _save_npz_once(audit_root / APP_BUNDLE_REL, {
            "record_ids": ids,
            "present_stream": present.astype(bool),
            "p_a02": p_a02.astype(np.float64),
            "p_pos_original": policies["probability_pos_original"].astype(np.float64),
            "p_pos_seed17": H.load_pilot_helpers().offset_probability(position, offset, gamma17),
            "p_pos_seed137": H.load_pilot_helpers().offset_probability(position, offset, gamma137),
            "probability_champion": policies["probability_champion"].astype(np.float64),
            "probability_candidate": policies["probability_candidate"].astype(np.float64),
            "labels_champion": champion_labels.astype(np.int8),
            "labels_candidate": candidate_labels.astype(np.int8),
        })
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["record_id", "artifact_signal"])
            writer.writerows(zip(ids.tolist(), candidate_labels.astype(int).tolist()))
        resource_after = _resource_snapshot("after single authorized E79 Test application")
        _require_resource(resource_after, POST_FIT_RESERVE_BYTES, "E79 post-application reserve")
        receipt_body = {
            "schema_version": 1,
            "candidate_id": CANDIDATE_ID,
            "status": "PASS_E79_APPLICATION_CSV_CREATED",
            "mode": "official",
            "source_commit": _git(audit_root, "rev-parse", "HEAD"),
            "freeze_signature": freeze_check["freeze_signature"],
            "root_pass_sha256": root_pass["pass_sha256"],
            "root_pass_file_sha256": root_pass["pass_file_sha256"],
            "fit_receipt_file_sha256": {
                H.fit_id_for_seed(seed): sha256_file(
                    _fit_paths(audit_root, H.fit_id_for_seed(seed))[2])
                for seed in H.SEEDS
            },
            "application_input_hashes": bindings,
            "official_test_rows": len(test),
            "official_test_cycles": [80, 91],
            "ordered_test_ids_sha256": H.sha256_array(ids),
            "test_feature_columns_requested": columns,
            "p_a02_sha256": H.sha256_array(p_a02),
            "position_matrix_logical_sha256": gamma_runner.positional.sha256_csr(position),
            "offset_sha256": H.sha256_array(offset),
            "probability_champion_sha256": H.sha256_array(policies["probability_champion"]),
            "probability_candidate_sha256": H.sha256_array(policies["probability_candidate"]),
            "labels_champion_sha256": H.sha256_array(champion_labels),
            "labels_candidate_sha256": H.sha256_array(candidate_labels),
            "baseline_champion_decision_parity": True,
            "baseline_rows": len(baseline_ids),
            "baseline_csv_sha256": bindings["baseline_csv_sha256"],
            "missing_stream_a02_fallback_bit_exact": bool(
                np.array_equal(policies["probability_candidate"][missing], p_a02[missing])),
            "present_threshold": H.PRESENT_THRESHOLD,
            "missing_threshold": H.MISSING_THRESHOLD,
            "threshold_operator": ">=",
            "candidate_csv_path": CSV_REL,
            "candidate_csv_sha256": sha256_file(csv_path),
            "candidate_csv_columns": ["record_id", "artifact_signal"],
            "candidate_csv_rows": len(ids),
            "application_bundle_path": APP_BUNDLE_REL,
            "application_bundle_sha256": bundle_sha,
            "fit_count_during_application": 0,
            "inference_count": 1,
            "test_targets_requested_or_loaded": False,
            "sample_submission_opened": False,
            "kaggle_contacted": False,
            "resource_before": resource_before,
            "resource_after": resource_after,
        }
        _write_json_once(receipt_path, receipt_body)
    return verifier.verify_application(output, receipt_path=receipt_path,
                                       bundle_path=bundle_path, csv_path=csv_path,
                                       baseline_csv_path=repo / BASELINE_CSV_REL,
                                       expected_baseline_sha256=EXPECTED_BASELINE_CSV_SHA256)


def _synthetic_inputs() -> dict[str, Any]:
    rng = np.random.default_rng(20261106)
    n_train, n_test, width = 144, 36, 24
    dense = rng.normal(size=(n_train, width))
    dense[rng.random(dense.shape) < 0.72] = 0.0
    x_train = sparse.csr_matrix(dense, dtype=np.float64)
    offset_train = rng.normal(scale=0.7, size=n_train)
    original_gamma = rng.normal(scale=0.05, size=width).astype(np.float64)
    p_train = H.load_pilot_helpers().offset_probability(x_train, offset_train, original_gamma)
    y_train = (rng.random(n_train) < p_train).astype(np.int8)
    test_dense = rng.normal(size=(n_test, width))
    test_dense[rng.random(test_dense.shape) < 0.72] = 0.0
    x_test = sparse.csr_matrix(test_dense, dtype=np.float64)
    offset_test = rng.normal(scale=0.7, size=n_test)
    present = np.ones(n_test, dtype=bool)
    present[[2, 9, 17, 31]] = False
    p_a02 = np.linspace(0.15, 0.85, n_test, dtype=np.float64)
    ids = np.asarray([f"SYNTH-{i:04d}" for i in range(n_test)], dtype=str)
    return {"x_train": x_train, "offset_train": offset_train, "y_train": y_train,
            "original_gamma": original_gamma, "x_test": x_test, "offset_test": offset_test,
            "present": present, "p_a02": p_a02, "ids": ids}


def run_synthetic(output_dir: Path) -> dict[str, Any]:
    """Run the lifecycle with only deterministic synthetic arrays and files."""
    verifier = _load_verifier(AUDIT_ROOT)
    output_dir = output_dir.resolve()
    inputs = _synthetic_inputs()
    input_path = output_dir / "SYNTHETIC_INPUTS.npz"
    if not input_path.exists():
        _save_npz_once(input_path, {"x_train_data": inputs["x_train"].data,
                                   "x_train_indices": inputs["x_train"].indices,
                                   "x_train_indptr": inputs["x_train"].indptr,
                                   "x_train_shape": np.asarray(inputs["x_train"].shape, dtype=np.int64),
                                   "offset_train": inputs["offset_train"], "y_train": inputs["y_train"],
                                   "original_gamma": inputs["original_gamma"],
                                   "x_test_data": inputs["x_test"].data,
                                   "x_test_indices": inputs["x_test"].indices,
                                   "x_test_indptr": inputs["x_test"].indptr,
                                   "x_test_shape": np.asarray(inputs["x_test"].shape, dtype=np.int64),
                                   "offset_test": inputs["offset_test"], "present": inputs["present"],
                                   "p_a02": inputs["p_a02"], "record_ids": inputs["ids"]})
    fit_root = output_dir / "fits"
    ledger_path = output_dir / "synthetic_fit_ledger.jsonl"
    fit_receipts: dict[str, dict[str, Any]] = {}
    for seed in H.SEEDS:
        fit_id = H.fit_id_for_seed(seed)
        fit_dir = fit_root / fit_id
        attempt_path, gamma_path, receipt_path = (fit_dir / "ATTEMPT.json", fit_dir / "gamma.npz",
                                                   fit_dir / "FIT_RECEIPT.json")
        if receipt_path.exists() or gamma_path.exists() or attempt_path.exists():
            H.require(attempt_path.exists() and gamma_path.exists() and receipt_path.exists(),
                      f"Synthetic partial one-shot fit; no retry: {fit_id}")
            _gamma, receipt = verifier.verify_fit_artifact(fit_root, fit_id,
                                                           freeze_signature="SYNTHETIC_FREEZE")
            fit_receipts[fit_id] = receipt
            continue
        raw, weights = _load_poisson_weights(inputs["x_train"].shape[0], seed)
        attempt = {"status": "SYNTHETIC_FIT_STARTED", "fit_id": fit_id, "seed": seed,
                   "attempt_number": 1, "test_or_sample_access": False}
        _write_once(attempt_path, (json.dumps(attempt, sort_keys=True) + "\n").encode())
        _append_event(ledger_path, {"event": "FIT_STARTED", "fit_id": fit_id, "seed": seed,
                                    "fit_rows": inputs["x_train"].shape[0], "synthetic": True})
        gamma, optimizer = H.load_pilot_helpers().fit_poisson_weighted_offset(
            inputs["x_train"], inputs["offset_train"], inputs["y_train"], weights)
        artifact_sha = _save_npz_once(gamma_path, {"gamma": gamma})
        receipt = {
            "schema_version": 1, "candidate_id": CANDIDATE_ID, "fit_id": fit_id,
            "seed": seed, "status": "PASS_E79_POISSON_GAMMA_FIT_PERSISTED", "mode": "synthetic",
            "source_commit": "SYNTHETIC", "freeze_signature": "SYNTHETIC_FREEZE",
            "fit_population": {"rows": len(inputs["y_train"]), "cycles": [0, 0]},
            "input_hashes": {"x": H.sha256_array(inputs["x_train"].data),
                             "offset": H.sha256_array(inputs["offset_train"]),
                             "target": H.sha256_array(inputs["y_train"])},
            "poisson_lambda": 1.0, "raw_counts_sha256": H.sha256_array(raw),
            "normalized_weights_sha256": H.sha256_array(weights),
            "normalized_weights_sum": float(weights.sum()),
            "objective": "sum_i a_i*logaddexp(0,-(2*y_i-1)*(z0_i+Xpos_i@gamma)) + 0.5*||gamma||_2^2",
            "ridge_c": 1.0, "global_strength_penalty_used": False,
            "optimizer": optimizer, "gamma_shape": list(gamma.shape),
            "gamma_dtype": str(gamma.dtype), "gamma_sha256": H.sha256_array(gamma),
            "artifact_path": f"fits/{fit_id}/gamma.npz", "artifact_sha256": artifact_sha,
            "fit_count": 1, "retry_allowed": False, "test_or_sample_access": False,
        }
        _write_json_once(receipt_path, receipt)
        _append_event(ledger_path, {"event": "FIT_SAVED", "fit_id": fit_id,
                                    "artifact_sha256": artifact_sha, "synthetic": True})
        _gamma, verified_receipt = verifier.verify_fit_artifact(fit_root, fit_id,
                                                                freeze_signature="SYNTHETIC_FREEZE")
        fit_receipts[fit_id] = verified_receipt
    fit_qa = verifier.verify_fit_set(fit_root, freeze_signature="SYNTHETIC_FREEZE")
    gammas = {seed: verifier.verify_fit_artifact(fit_root, H.fit_id_for_seed(seed),
                                                 freeze_signature="SYNTHETIC_FREEZE")[0]
              for seed in H.SEEDS}
    policies = H.build_probabilities(inputs["x_test"], inputs["offset_test"],
                                     inputs["original_gamma"], gammas[17], gammas[137],
                                     inputs["p_a02"], inputs["present"])
    labels_champ = H.apply_thresholds(policies["probability_champion"], inputs["present"])
    labels_candidate = H.apply_thresholds(policies["probability_candidate"], inputs["present"])
    baseline_path = output_dir / "SYNTHETIC_BASELINE.csv"
    if not baseline_path.exists():
        with baseline_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["record_id", "artifact_signal"])
            writer.writerows(zip(inputs["ids"].tolist(), labels_champ.astype(int).tolist()))
    bundle_path = output_dir / "synthetic_application_predictions.npz"
    csv_path = output_dir / "synthetic_submission.csv"
    receipt_path = output_dir / "synthetic_application_receipt.json"
    if not receipt_path.exists():
        bundle_sha = _save_npz_once(bundle_path, {
            "record_ids": inputs["ids"], "present_stream": inputs["present"],
            "p_a02": inputs["p_a02"],
            "p_pos_original": policies["probability_pos_original"],
            "p_pos_seed17": H.load_pilot_helpers().offset_probability(inputs["x_test"], inputs["offset_test"], gammas[17]),
            "p_pos_seed137": H.load_pilot_helpers().offset_probability(inputs["x_test"], inputs["offset_test"], gammas[137]),
            "probability_champion": policies["probability_champion"],
            "probability_candidate": policies["probability_candidate"],
            "labels_champion": labels_champ, "labels_candidate": labels_candidate,
        })
        with csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["record_id", "artifact_signal"])
            writer.writerows(zip(inputs["ids"].tolist(), labels_candidate.astype(int).tolist()))
        receipt = {
            "schema_version": 1, "candidate_id": CANDIDATE_ID,
            "status": "PASS_E79_APPLICATION_CSV_CREATED", "mode": "synthetic",
            "freeze_signature": "SYNTHETIC_FREEZE", "source_commit": "SYNTHETIC",
            "application_input_hashes": {"test_features": H.sha256_array(inputs["offset_test"]),
                                          "p_a02": H.sha256_array(inputs["p_a02"])},
            "official_test_rows": len(inputs["ids"]),
            "ordered_test_ids_sha256": H.sha256_array(inputs["ids"]),
            "fit_receipt_file_sha256": {
                fit_id: sha256_file(fit_root / fit_id / "FIT_RECEIPT.json")
                for fit_id in H.FIT_IDS
            },
            "baseline_champion_decision_parity": True,
            "baseline_csv_sha256": sha256_file(baseline_path),
            "missing_stream_a02_fallback_bit_exact": bool(np.array_equal(
                policies["probability_candidate"][~inputs["present"]],
                inputs["p_a02"][~inputs["present"]])),
            "present_threshold": H.PRESENT_THRESHOLD, "missing_threshold": H.MISSING_THRESHOLD,
            "threshold_operator": ">=", "candidate_csv_path": csv_path.name,
            "candidate_csv_sha256": sha256_file(csv_path),
            "candidate_csv_columns": ["record_id", "artifact_signal"],
            "candidate_csv_rows": len(inputs["ids"]),
            "application_bundle_path": bundle_path.name,
            "application_bundle_sha256": bundle_sha,
            "fit_count_during_application": 0, "inference_count": 1,
            "test_targets_requested_or_loaded": False,
            "sample_submission_opened": False, "kaggle_contacted": False,
        }
        _write_json_once(receipt_path, receipt)
    app_qa = verifier.verify_application(fit_root, receipt_path=receipt_path,
                                         bundle_path=bundle_path, csv_path=csv_path,
                                         baseline_csv_path=baseline_path,
                                         expected_baseline_sha256=sha256_file(baseline_path))
    ledger = _ledger_events(ledger_path)
    H.require(sum(e.get("event") == "FIT_STARTED" for e in ledger) == 2
              and sum(e.get("event") == "FIT_SAVED" for e in ledger) == 2,
              "Synthetic resume changed the exact two-fit ledger")
    return {"status": "PASS_SYNTHETIC_E79_END_TO_END", "fit_count": 2,
            "application_count": 1, "fit_qa": fit_qa, "application_qa": app_qa,
            "ledger_fit_starts": 2, "ledger_fit_saves": 2,
            "test_or_sample_access": False, "kaggle_contacted": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True,
                        choices=("synthetic", "freeze", "verify-freeze", "reserve", "fit-all", "apply-test"))
    parser.add_argument("--repository", type=Path,
                        help="Explicit official TP2 repository path; required outside synthetic mode")
    parser.add_argument("--output-dir", type=Path,
                        help="Isolated synthetic output directory; required for synthetic mode")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            H.require(args.output_dir is not None, "--output-dir is required for synthetic mode")
            result = run_synthetic(args.output_dir)
        else:
            H.require(args.repository is not None, "--repository is required for official commands")
            if args.command == "freeze":
                result = create_source_freeze(args.repository)
            elif args.command == "verify-freeze":
                result = verify_source_freeze(args.repository)
            elif args.command == "reserve":
                result = reserve_fits(args.repository)
            elif args.command == "fit-all":
                result = fit_all(args.repository)
            else:
                result = apply_test_once(args.repository)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True, ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
