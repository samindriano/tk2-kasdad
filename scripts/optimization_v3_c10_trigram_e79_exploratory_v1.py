"""One-shot full-Train C10 trigram benchmark; never uploads to Kaggle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_round2_models import stream_missing_mask
from src.models.classification_models import _text_values
from src.models.optimization_v3_batch1_c10 import (
    CANDIDATE_WEIGHT,
    C2_WEIGHT,
    FEATURE_NAMES,
    SPARSE_PARAMS,
    THRESHOLD,
    TRIGRAM_PARAMS,
    SeparateTrigramSparseClassifier,
    _normalise_parameters,
    _ordered_vocabulary_digest,
    _select_features,
    trigram_vectorizer,
    word_vectorizer,
)
from src.models.regression_research_text import RegressionTextModel
from src.models.round1_evaluation import array_digest

REPORT = ROOT / "reports/optimization_v3/c10_trigram_e79_exploratory_v1"
RUN = ROOT / "experiments/runs/optimization_v3_c10_trigram_e79_exploratory_v1"
SUBMISSIONS = ROOT / "submissions/optimization_v3/c10_trigram_e79_exploratory_v1"
CSV_PATH = SUBMISSIONS / "c10_trigram_e79_exploratory.csv"
TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
TEST_PATH = ROOT / "data/raw/classification/Test.csv"
RECIPE_PATH = REPORT / "RECIPE.json"
PREFIT_AUDIT_PATH = REPORT / "PREFIT_AUDIT.json"
PREFIT_FREEZE_PATH = REPORT / "PRE_FIT_FREEZE.json"
SYNTHETIC_QA_PATH = REPORT / "SYNTHETIC_QA.json"
ARMED_PATH = REPORT / "FIT_ARMED.json"
LEDGER_PATH = REPORT / "BUDGET_LEDGER.jsonl"
FIT_RECEIPT_PATH = REPORT / "FIT_RECEIPT.json"
TRAINED_FREEZE_PATH = REPORT / "TRAINED_STATE_FREEZE.json"
INFERENCE_BUILD_PATH = REPORT / "INFERENCE_BUILD.json"
PREDICTIONS_PATH = RUN / "test_predictions.npz"
MODEL_PATH = RUN / "c10_model.joblib"

OLD_RECIPE = ROOT / "experiments/optimization_v3/batch1/C10/recipe.json"
OLD_FREEZE = ROOT / "experiments/optimization_v3/batch1/C10/freeze.json"
OLD_RESULTS = ROOT / "reports/optimization_v3/batch1/C10/results.json"
OLD_VERIFICATION = ROOT / "reports/optimization_v3/batch1/C10/verification.json"
OLD_PREFLIGHT = ROOT / "reports/optimization_v3/batch1/C10/preflight.json"
OLD_RUN = ROOT / "experiments/runs/optimization_v3_batch1/C10"
SPARSE25_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
SPARSE25_BUNDLE = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"

FIT_ID = "C10_TRIGRAM_E79_EXPLORATORY_V1_FULLTRAIN_LGBM"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_OLD_RECIPE_FILE_SHA = "491f005e45168310481b72362baa5cfeb857bc1501c7459e61f16609cedc5ba8"
EXPECTED_OLD_FREEZE_FILE_SHA = "dfd7bfb65bf44f370603f4e6029fe19d8941b7fecb91d50831b70c6f5ef57a62"
EXPECTED_OLD_RESULTS_SHA = "83b5be211558a58a7abf3fe1161daf3f9868d37e979481eeafa6f5635cf54003"
EXPECTED_OLD_VERIFICATION_SHA = "9fcfb22e45874182de4226b1a1233c7f360d72dc4f1877522151d1acae1152dd"
EXPECTED_BASELINE_MANIFEST_SHA = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
EXPECTED_BASELINE_BUNDLE_SHA = "31ef6945ae74a14b1e4ff46220667ac3a08de3401120257e7932745c927d6103"
GIB = 1024 ** 3
MAX_UNITS = 6
MAX_RSS_BYTES = 6 * GIB
MIN_AVAILABLE_BYTES = 4 * GIB
MIN_PROJECTED_RESERVE_BYTES = 2 * GIB
SOURCE_RELATIVE = (
    "scripts/optimization_v3_c10_trigram_e79_exploratory_v1.py",
    "scripts/verify_optimization_v3_c10_trigram_e79_exploratory_v1.py",
    "tests/test_optimization_v3_c10_trigram_e79_exploratory_v1.py",
    "src/models/optimization_v3_batch1_c10.py",
    "src/models/optimization_v3_sparse_classification.py",
    "src/models/classification_models.py",
    "src/models/regression_research_text.py",
    "src/models/regression_round2_models.py",
    "src/models/classification_round2_models.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/utils/paths.py",
)


class BenchmarkError(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != value:
            raise BenchmarkError(f"Refusing to overwrite different owned artifact: {path}")
        return
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"Invalid budget ledger JSON at line {number}") from exc
        if not isinstance(event, dict):
            raise BenchmarkError(f"Budget ledger row {number} is not an object")
        result.append(event)
    return result


def validate_budget_events(events: list[dict[str, Any]], *, max_units: int = MAX_UNITS) -> None:
    """Fail closed on duplicate fit IDs, retries, over-cap reservations, or bad order."""
    if not events:
        raise BenchmarkError("C10 budget ledger is empty; the six-unit reservation is missing")
    if any(event.get("fit_id") != FIT_ID for event in events):
        raise BenchmarkError("C10 isolated ledger contains a foreign or missing fit ID")
    kinds = [event.get("event") for event in events]
    allowed = ("FIT_RESERVED", "FIT_ARMED", "FIT_STARTED", "FIT_SAVED", "FIT_FAILED")
    if any(kind not in allowed for kind in kinds):
        raise BenchmarkError("Unknown event in isolated C10 budget ledger")
    valid_prefixes = (
        ["FIT_RESERVED"],
        ["FIT_RESERVED", "FIT_ARMED"],
        ["FIT_RESERVED", "FIT_ARMED", "FIT_STARTED"],
        ["FIT_RESERVED", "FIT_ARMED", "FIT_STARTED", "FIT_SAVED"],
        ["FIT_RESERVED", "FIT_ARMED", "FIT_STARTED", "FIT_FAILED"],
    )
    if kinds not in valid_prefixes:
        raise BenchmarkError("C10 budget ledger is out of order, duplicated, or retried")
    if int(events[0].get("units", 0)) != max_units:
        raise BenchmarkError("C10 reservation differs from its six-unit cap")
    if kinds[-1] == "FIT_SAVED" and int(events[-1].get("units", 0)) != max_units:
        raise BenchmarkError("C10 saved fit does not reconcile to the reserved six units")
    if kinds[-1] == "FIT_FAILED" and int(events[-1].get("retry_count", -1)) != 0:
        raise BenchmarkError("C10 failed attempt reports a retry")


def append_event(event: dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = _read_events()
    validate_budget_events([*current, event])
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    fd = os.open(LEDGER_PATH, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _source_hashes() -> dict[str, str]:
    missing = [rel for rel in SOURCE_RELATIVE if not (ROOT / rel).is_file()]
    if missing:
        raise BenchmarkError(f"Required source files are missing: {missing}")
    return {rel: file_sha(ROOT / rel) for rel in SOURCE_RELATIVE}


def _committed_source_hashes(commit: str) -> dict[str, str]:
    """Verify every code input is present at the pinned commit with current bytes."""
    verified: dict[str, str] = {}
    for relative_path in SOURCE_RELATIVE:
        show = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}:{relative_path}"],
                              capture_output=True, check=False)
        if show.returncode:
            raise BenchmarkError(f"C10 source is absent from its claimed commit: {relative_path}")
        diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, "--", relative_path],
                              check=False)
        if diff.returncode != 0:
            raise BenchmarkError(f"C10 source differs from its claimed commit: {relative_path}")
        verified[relative_path] = file_sha(ROOT / relative_path)
    return verified


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib", "psutil")
    return {name: importlib.metadata.version(name) for name in names}


def _historical_source_audit(manifest: dict[str, Any]) -> dict[str, str]:
    commit = manifest.get("source_commit")
    source_map = manifest.get("source_code_sha256")
    if not isinstance(commit, str) or not isinstance(source_map, dict) or not source_map:
        raise BenchmarkError("Sparse25 manifest lacks its historical source pins")
    verified: dict[str, str] = {}
    for relative_path, expected in sorted(source_map.items()):
        result = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative_path}"],
                                capture_output=True, check=False)
        if result.returncode:
            raise BenchmarkError(f"Cannot resolve frozen Sparse25 source blob: {relative_path}")
        actual = hashlib.sha256(result.stdout).hexdigest()
        if actual != expected:
            raise BenchmarkError(f"Historical Sparse25 source hash mismatch: {relative_path}")
        verified[relative_path] = actual
    return verified


def _fit_component_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for item in manifest.get("component_audits", []):
        component = item.get("fit_component") or {}
        name = component.get("model")
        if name in components:
            raise BenchmarkError(f"Sparse25 manifest repeats component: {name}")
        if name:
            components[str(name)] = item
    return components


def _resource_estimate() -> dict[str, int]:
    old = _read_json(OLD_PREFLIGHT)["resource_forecast"]
    fit_rows_old = int(old["largest_fit_rows"])
    estimate_old = int(old["estimated_additional_process_bytes"])
    train_rows = 50608
    estimated_peak = int(old["process_rss_bytes"] + estimate_old * train_rows / fit_rows_old)
    available = int(psutil.virtual_memory().available)
    return {
        "available_ram_bytes": available,
        "estimated_process_peak_bytes": estimated_peak,
        "projected_free_after_peak_bytes": available - estimated_peak,
        "minimum_available_ram_bytes": MIN_AVAILABLE_BYTES,
        "minimum_projected_reserve_bytes": MIN_PROJECTED_RESERVE_BYTES,
        "process_rss_cap_bytes": MAX_RSS_BYTES,
    }


def _check_resource_budget() -> dict[str, int]:
    forecast = _resource_estimate()
    if forecast["available_ram_bytes"] < MIN_AVAILABLE_BYTES:
        raise BenchmarkError("NO_GO_RESOURCE: free RAM is below 4 GiB")
    if forecast["projected_free_after_peak_bytes"] < MIN_PROJECTED_RESERVE_BYTES:
        raise BenchmarkError("NO_GO_RESOURCE: projected RAM reserve is below 2 GiB")
    if forecast["estimated_process_peak_bytes"] >= MAX_RSS_BYTES:
        raise BenchmarkError("NO_GO_RESOURCE: estimated process peak reaches the 6 GiB RSS cap")
    return forecast


def _active_training_processes() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["pid"] == current_pid:
                continue
            name = (process.info["name"] or "").casefold()
            command = " ".join(process.info["cmdline"] or []).casefold()
            if ("python" in name or name.endswith("python.exe")) and any(
                marker in command for marker in ("lightgbm", "catboost", "optimization_v3", "pytest")
            ):
                found.append({"pid": process.info["pid"], "name": process.info["name"],
                              "command": command[:240]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _target_free_train_metadata() -> dict[str, Any]:
    if not TRAIN_PATH.is_file() or file_sha(TRAIN_PATH) != EXPECTED_TRAIN_SHA:
        raise BenchmarkError("Official Train hash mismatch or file unavailable")
    frame = pd.read_csv(TRAIN_PATH, usecols=["record_id", "cycle_code"])
    if len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise BenchmarkError("Target-free Train identity audit failed")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64).to_numpy()
    if (int(cycles.min()), int(cycles.max())) != (47, 79) or set(np.unique(cycles)) != set(range(47, 80)):
        raise BenchmarkError("Train cycle support is not the frozen 47-79 population")
    ids = frame["record_id"].astype(str).tolist()
    id_sha = hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()
    cycle_sha = hashlib.sha256(np.asarray(cycles, dtype="<i8").tobytes()).hexdigest()
    return {
        "train_sha256": EXPECTED_TRAIN_SHA,
        "rows": int(len(frame)),
        "cycles": [int(cycles.min()), int(cycles.max())],
        "cycle_counts": {str(int(c)): int((cycles == c).sum()) for c in sorted(set(cycles.tolist()))},
        "record_id_sequence_sha256": id_sha,
        "cycle_sequence_sha256": cycle_sha,
        "target_column_read": False,
    }


def run_prefit_audit() -> dict[str, Any]:
    if not REPORT.is_dir():
        raise BenchmarkError(f"Owned report namespace is missing: {REPORT}")
    if file_sha(OLD_RECIPE) != EXPECTED_OLD_RECIPE_FILE_SHA:
        raise BenchmarkError("Historical C10 recipe artifact changed")
    if file_sha(OLD_FREEZE) != EXPECTED_OLD_FREEZE_FILE_SHA:
        raise BenchmarkError("Historical C10 freeze artifact changed")
    if file_sha(OLD_RESULTS) != EXPECTED_OLD_RESULTS_SHA:
        raise BenchmarkError("Historical C10 result artifact changed")
    if file_sha(OLD_VERIFICATION) != EXPECTED_OLD_VERIFICATION_SHA:
        raise BenchmarkError("Historical C10 verifier artifact changed")
    old_recipe = _read_json(OLD_RECIPE)
    old_freeze = _read_json(OLD_FREEZE)
    old_results = _read_json(OLD_RESULTS)
    old_verification = _read_json(OLD_VERIFICATION)
    if (old_recipe.get("card_id") != "C10" or old_recipe.get("policy") != "C_SPARSE_REPLACE"
            or old_results.get("status") != "PILOT_COMPLETE_RETIRE"
            or old_verification.get("status") != "PASS"
            or old_results.get("freeze_signature") != old_verification.get("freeze_signature")):
        raise BenchmarkError("Historical C10 recipe/result/verifier lineage does not reconcile")
    if old_freeze.get("status") != "FROZEN_BEFORE_C10_PILOT_FITS":
        raise BenchmarkError("Historical C10 pilot freeze status changed")

    manifest = _read_json(SPARSE25_MANIFEST)
    if file_sha(SPARSE25_MANIFEST) != EXPECTED_BASELINE_MANIFEST_SHA:
        raise BenchmarkError("Sparse25 full-Train manifest hash mismatch")
    if file_sha(SPARSE25_BUNDLE) != EXPECTED_BASELINE_BUNDLE_SHA:
        raise BenchmarkError("Sparse25 full-Train inference bundle hash mismatch")
    if (manifest.get("status") != "PREPARED_AND_QA_PASS" or manifest.get("train_rows") != 50608
            or manifest.get("input_sha256", {}).get("train") != EXPECTED_TRAIN_SHA):
        raise BenchmarkError("Sparse25 full-Train provenance does not match official Train")
    if manifest.get("recipe") != {
        "kind": "availability_present_blend",
        "present": [{"model": "C2_B", "weight": 0.75}, {"model": "SPARSE_BINARY", "weight": 0.25}],
        "missing": "CB_MISSING", "threshold": 0.5,
    }:
        raise BenchmarkError("Sparse25 accepted full-Train policy differs from frozen C10 route")
    components = _fit_component_map(manifest)
    if set(components) != {"C2_B", "CB_MISSING", "SPARSE_BINARY"}:
        raise BenchmarkError("Sparse25 full-Train component set is incomplete or unexpected")
    component_summary = {}
    for name, item in components.items():
        if item.get("fit_rows") != 50608 or item.get("learned_only_official_Train") is not True:
            raise BenchmarkError(f"Sparse25 component does not have exact Train lineage: {name}")
        component_summary[name] = {
            "fit_rows": int(item["fit_rows"]),
            "state_sha256": str(item["state_sha256"]),
            "prediction_sha256": str(item["prediction_sha256"]),
            "learned_only_official_Train": True,
        }
    historical_sources = _historical_source_audit(manifest)
    with zipfile.ZipFile(SPARSE25_BUNDLE) as archive:
        bundle_members = sorted(name for name in archive.namelist() if name.endswith(".npy"))
    expected_members = sorted(f"{name}.npy" for name in
                              ("prediction", "probability", "base", "missing", "sparse", "missing_mask"))
    if bundle_members != expected_members:
        raise BenchmarkError("Sparse25 inference bundle array names changed; no arrays were opened")

    train_meta = _target_free_train_metadata()
    expected_run_names = {"C10_fit55.json", "C10_fit55.npz", "C10_fit67.json", "C10_fit67.npz", "attempts.jsonl"}
    old_run_names = {path.name for path in OLD_RUN.iterdir() if path.is_file()}
    if old_run_names != expected_run_names:
        raise BenchmarkError(f"Historical C10 run directory changed: {sorted(old_run_names)}")
    output_paths = [PREFIT_AUDIT_PATH, PREFIT_FREEZE_PATH, ARMED_PATH,
                    FIT_RECEIPT_PATH, TRAINED_FREEZE_PATH,
                    INFERENCE_BUILD_PATH, MODEL_PATH, PREDICTIONS_PATH, CSV_PATH]
    if any(path.exists() for path in output_paths):
        raise BenchmarkError("A prior C10 full-Train/E79 output exists; duplicate/retry blocked")

    try:
        recipe = _read_json(RECIPE_PATH)
    except FileNotFoundError as exc:
        raise BenchmarkError("New frozen C10 E79 recipe is missing") from exc
    if recipe.get("benchmark_id") != "C10_TRIGRAM_E79_EXPLORATORY_V1":
        raise BenchmarkError("New C10 recipe namespace mismatch")
    if tuple(FEATURE_NAMES) != tuple(recipe["model"]["feature_allowlist"]):
        raise BenchmarkError("Current C10 feature allowlist differs from signed recipe")
    if recipe["model"].get("class") != "src.models.optimization_v3_batch1_c10.SeparateTrigramSparseClassifier":
        raise BenchmarkError("C10 model class differs from the signed recipe")
    if _normalise_parameters(TRIGRAM_PARAMS) != recipe["model"]["separate_trigram_tfidf"]:
        raise BenchmarkError("Current C10 trigram vectorizer differs from signed recipe")
    frozen_lgbm = recipe["model"]["lightgbm"]
    current_lgbm = _normalise_parameters(SPARSE_PARAMS)
    if (any(current_lgbm.get(key) != value for key, value in frozen_lgbm.items()
            if key != "early_stopping")
            or frozen_lgbm.get("early_stopping") is not False):
        raise BenchmarkError("Current C10 classifier parameters differ from signed recipe")
    word_params = _normalise_parameters(word_vectorizer().get_params())
    required_word = recipe["model"]["word_tfidf"]
    if any(word_params.get(key) != value for key, value in required_word.items()):
        raise BenchmarkError("Current C10 word vectorizer differs from signed recipe")
    if (float(recipe["fixed_policy"].get("threshold", -1)) != THRESHOLD
            or float(C2_WEIGHT) != 0.75 or float(CANDIDATE_WEIGHT) != 0.25
            or recipe["fixed_policy"].get("present_stream") !=
            "0.75 * verified Sparse25 C2_B probability + 0.25 * new C10 raw probability"
            or recipe["fixed_policy"].get("missing_stream") !=
            "verified Sparse25 E79 CB_MISSING probability bit-exact"):
        raise BenchmarkError("C10 fixed blend, missing fallback, or threshold differs from signed recipe")
    budget_recipe = recipe["budget"]
    if (budget_recipe.get("fit_id") != FIT_ID
            or budget_recipe.get("max_learned_units") != MAX_UNITS
            or budget_recipe.get("official_predictor_fits") != 1
            or budget_recipe.get("retry_count") != 0):
        raise BenchmarkError("C10 budget cap or one-fit policy differs from signed recipe")

    versions = _package_versions()
    source_hashes = _source_hashes()
    head = _git("rev-parse", "HEAD")
    committed_sources = _committed_source_hashes(head)
    recipe_relative = RECIPE_PATH.relative_to(ROOT).as_posix()
    recipe_diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", head, "--", recipe_relative],
                                 check=False)
    if recipe_diff.returncode != 0:
        raise BenchmarkError("C10 recipe is not committed at the prefit source HEAD")
    audit = {
        "schema_version": 1,
        "status": "PASS_TARGET_FREE_C10_E79_PREFIT_AUDIT",
        "head_at_audit": head,
        "old_c10": {
            "recipe_sha256": file_sha(OLD_RECIPE),
            "freeze_sha256": file_sha(OLD_FREEZE),
            "results_sha256": file_sha(OLD_RESULTS),
            "verification_sha256": file_sha(OLD_VERIFICATION),
            "freeze_signature": old_results["freeze_signature"],
            "status": old_results["status"],
            "pilot_fit_archives": sorted(old_run_names),
            "fulltrain_c10_state_present": False,
        },
        "sparse25": {
            "manifest_sha256": file_sha(SPARSE25_MANIFEST),
            "bundle_sha256": file_sha(SPARSE25_BUNDLE),
            "source_commit": manifest["source_commit"],
            "historical_source_hashes_verified": historical_sources,
            "components": component_summary,
            "bundle_array_names_only_no_values_opened": bundle_members,
        },
        "official_train_target_free": train_meta,
        "new_code_hashes": source_hashes,
        "committed_code_hashes": committed_sources,
        "packages": versions,
        "test_features_opened": False,
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    return audit


def write_prefit_freeze() -> dict[str, Any]:
    if not SYNTHETIC_QA_PATH.is_file() or _read_json(SYNTHETIC_QA_PATH).get("status") != "PASS_SYNTHETIC_C10_END_TO_END_AND_RESUME":
        raise BenchmarkError("Synthetic end-to-end and persistence/resume QA must pass first")
    if _read_events():
        raise BenchmarkError("C10 isolated budget ledger already exists; duplicate reservation blocked")
    if any(path.exists() for path in (PREFIT_AUDIT_PATH, PREFIT_FREEZE_PATH, ARMED_PATH,
                                     FIT_RECEIPT_PATH, TRAINED_FREEZE_PATH,
                                     INFERENCE_BUILD_PATH, MODEL_PATH, PREDICTIONS_PATH, CSV_PATH)):
        raise BenchmarkError("Partial or prior C10 full-Train/E79 output exists; duplicate/retry blocked")
    audit = run_prefit_audit()
    if audit["status"] != "PASS_TARGET_FREE_C10_E79_PREFIT_AUDIT":
        raise BenchmarkError("Target-free prefit audit did not pass")
    if PREFIT_AUDIT_PATH.exists():
        if _read_json(PREFIT_AUDIT_PATH) != audit:
            raise BenchmarkError("Existing C10 prefit audit differs; no overwrite")
    else:
        _write_json_once(PREFIT_AUDIT_PATH, audit)
    recipe_sha = file_sha(RECIPE_PATH)
    body = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_C10_E79_FULLTRAIN_FIT",
        "benchmark_id": "C10_TRIGRAM_E79_EXPLORATORY_V1",
        "fit_id": FIT_ID,
        "source_commit": _git("rev-parse", "HEAD"),
        "recipe_sha256": recipe_sha,
        "prefit_audit_sha256": file_sha(PREFIT_AUDIT_PATH),
        "synthetic_qa_sha256": file_sha(SYNTHETIC_QA_PATH),
        "source_hashes": audit["new_code_hashes"],
        "train": audit["official_train_target_free"],
        "baseline": audit["sparse25"],
        "fit_population": {"rows": 50608, "cycle_min": 47, "cycle_max": 79,
                           "target": "artifact_signal", "feature_fit_source": "official Train only"},
        "budget": {"cap_units": MAX_UNITS, "reserved_units": MAX_UNITS,
                   "predictor_fits": 1, "max_preprocessing_fits": 5,
                   "retries": 0, "reservation_id": FIT_ID},
        "resource_limits": {
            "jobs": 1, "threads": 2,
            "minimum_initial_free_ram_bytes": MIN_AVAILABLE_BYTES,
            "minimum_projected_free_ram_bytes": MIN_PROJECTED_RESERVE_BYTES,
            "maximum_process_rss_bytes": MAX_RSS_BYTES,
        },
        "test_features_opened": False,
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    freeze = {**body, "freeze_signature": canonical_sha(body)}
    _write_json_once(PREFIT_FREEZE_PATH, freeze)
    reservation = {
        "event": "FIT_RESERVED", "fit_id": FIT_ID,
        "units": MAX_UNITS, "predictor_fits": 1, "preprocessing_units_max": 5,
        "retry_count": 0, "source_commit": body["source_commit"],
        "recipe_sha256": recipe_sha, "freeze_signature": freeze["freeze_signature"],
    }
    append_event(reservation)
    return freeze


def arm_fit() -> dict[str, Any]:
    freeze = _read_json(PREFIT_FREEZE_PATH)
    signature = freeze.pop("freeze_signature", None)
    if signature != canonical_sha(freeze):
        raise BenchmarkError("C10 prefit freeze signature mismatch")
    freeze["freeze_signature"] = signature
    if _git("rev-parse", "HEAD") == freeze["source_commit"]:
        raise BenchmarkError("Commit the prefit freeze and budget reservation before fit arm")
    if _source_hashes() != freeze["source_hashes"]:
        raise BenchmarkError("C10 source changed after the prefit freeze")
    _committed_source_hashes(freeze["source_commit"])
    events = _read_events()
    validate_budget_events(events)
    if any(event.get("event") != "FIT_RESERVED" for event in events):
        raise BenchmarkError("C10 fit is already armed, started, or terminal; no retry")
    if _active_training_processes():
        raise BenchmarkError("Another model training job is active")
    resource = _check_resource_budget()
    armed = {
        "schema_version": 1,
        "status": "PASS_ARMED_ONE_C10_FULLTRAIN_FIT",
        "fit_id": FIT_ID,
        "armed_at_head": _git("rev-parse", "HEAD"),
        "source_commit": freeze["source_commit"],
        "freeze_signature": signature,
        "resource_snapshot": resource,
        "jobs": 1,
        "threads": 2,
        "concurrent_training_jobs": 0,
    }
    _write_json_once(ARMED_PATH, armed)
    append_event({"event": "FIT_ARMED", "fit_id": FIT_ID,
                  "head": armed["armed_at_head"], "freeze_signature": signature,
                  "resource_snapshot": resource, "jobs": 1, "threads": 2})
    return armed


def _git_file_commit(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", relative],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def fit_full_train() -> dict[str, Any]:
    freeze = _read_json(PREFIT_FREEZE_PATH)
    signature = freeze.pop("freeze_signature", None)
    if signature != canonical_sha(freeze):
        raise BenchmarkError("C10 prefit freeze signature mismatch")
    freeze["freeze_signature"] = signature
    armed = _read_json(ARMED_PATH)
    if armed.get("status") != "PASS_ARMED_ONE_C10_FULLTRAIN_FIT" or armed.get("fit_id") != FIT_ID:
        raise BenchmarkError("C10 fit has no valid committed arm")
    head = _git("rev-parse", "HEAD")
    if _git_file_commit(ARMED_PATH) != head:
        raise BenchmarkError("Commit FIT_ARMED.json before starting the official fit")
    if _source_hashes() != freeze["source_hashes"]:
        raise BenchmarkError("C10 source changed after its committed prefit freeze")
    _committed_source_hashes(freeze["source_commit"])
    if any(path.exists() for path in (MODEL_PATH, FIT_RECEIPT_PATH, TRAINED_FREEZE_PATH)):
        raise BenchmarkError("C10 full-Train fit output already exists; no retry")
    if _active_training_processes():
        raise BenchmarkError("Another model training job is active")
    resource = _check_resource_budget()

    frame = pd.read_csv(TRAIN_PATH)
    if file_sha(TRAIN_PATH) != EXPECTED_TRAIN_SHA or len(frame) != 50608:
        raise BenchmarkError("Official Train changed or has wrong fit population")
    if "artifact_signal" not in frame.columns:
        raise BenchmarkError("Official Train target column is missing")
    target_float = pd.to_numeric(frame.pop("artifact_signal"), errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target_float).all() or not np.isin(target_float, (0.0, 1.0)).all():
        raise BenchmarkError("Official Train artifact_signal is not finite binary data")
    target = target_float.astype(np.int8)
    x = _select_features(frame.loc[:, list(FEATURE_NAMES)])
    cycles = pd.to_numeric(x["cycle_code"], errors="raise").astype(np.int64).to_numpy()
    if set(np.unique(cycles)) != set(range(47, 80)) or len(target) != len(x):
        raise BenchmarkError("Official Train is not the frozen full cycle 47-79 fit population")
    if ("record_id" not in frame or frame["record_id"].isna().any()
            or frame["record_id"].duplicated().any()):
        raise BenchmarkError("Official Train record_id is missing or duplicated")
    if _git("rev-parse", "HEAD") != head:
        raise BenchmarkError("Git HEAD changed while preparing the fit")

    started = {
        "event": "FIT_STARTED", "fit_id": FIT_ID, "head": head,
        "source_commit": freeze["source_commit"], "freeze_signature": signature,
        "train_sha256": EXPECTED_TRAIN_SHA, "fit_rows": int(len(target)),
        "cycle_range": [int(cycles.min()), int(cycles.max())],
        "target_sha256": array_digest(target.astype(np.int8)),
        "resource_snapshot": resource, "jobs": 1, "threads": 2,
    }
    events = _read_events()
    validate_budget_events(events)
    if any(event.get("event") != "FIT_RESERVED" and event.get("event") != "FIT_ARMED" for event in events):
        raise BenchmarkError("C10 fit start already exists; no retry")
    append_event(started)

    model = SeparateTrigramSparseClassifier()
    started_at = time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            model.fit(x, target)
        if model.audit_state().get("fit_rows") != 50608:
            raise BenchmarkError("Fitted C10 estimator state does not show all 50,608 Train rows")
        RUN.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            raise BenchmarkError("C10 model artifact already exists; no overwrite")
        temp_path = RUN / "c10_model.joblib.partial"
        if temp_path.exists():
            raise BenchmarkError("C10 partial model artifact already exists; no retry")
        joblib.dump(model, temp_path, compress=0)
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, MODEL_PATH)
        model_sha = file_sha(MODEL_PATH)
        elapsed = time.perf_counter() - started_at
        state = model.audit_state()
        receipt_body = {
            "schema_version": 1,
            "status": "PASS_ONE_C10_FULLTRAIN_PREDICTOR_FIT_SAVED",
            "fit_id": FIT_ID,
            "source_commit": freeze["source_commit"],
            "fit_start_head": head,
            "recipe_sha256": freeze["recipe_sha256"],
            "prefit_freeze_signature": signature,
            "train_sha256": EXPECTED_TRAIN_SHA,
            "fit_rows": int(len(target)),
            "cycle_range": [int(cycles.min()), int(cycles.max())],
            "cycle_counts": {str(int(cycle)): int(np.count_nonzero(cycles == cycle))
                             for cycle in sorted(np.unique(cycles))},
            "target_counts": {str(int(k)): int((target == k).sum()) for k in (0, 1)},
            "target_sha256": array_digest(target.astype(np.int8)),
            "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
            "model_file_sha256": model_sha,
            "model_state": state,
            "model_state_sha256": state["state_sha256"],
            "official_predictor_fits": 1,
            "preprocessing_units": {
                "structured_selected_preprocessor": 1,
                "standard_scaler": 1,
                "one_hot_encoder": 1,
                "word_tfidf": 1,
                "trigram_tfidf": 1,
            },
            "learned_unit_total": 6,
            "duration_seconds": float(elapsed),
            "test_features_opened": False,
            "test_labels_opened": False,
            "sample_submission_opened": False,
            "kaggle_contacted": False,
            "resource_snapshot_at_fit_start": resource,
        }
        receipt = {**receipt_body, "receipt_sha256": canonical_sha(receipt_body)}
        _write_json_once(FIT_RECEIPT_PATH, receipt)
        trained_body = {
            "schema_version": 1,
            "status": "TRAINED_STATE_FROZEN_BEFORE_TEST_FEATURES",
            "fit_id": FIT_ID,
            "source_commit": freeze["source_commit"],
            "fit_start_head": head,
            "train_sha256": EXPECTED_TRAIN_SHA,
            "fit_receipt_sha256": file_sha(FIT_RECEIPT_PATH),
            "model_file_sha256": model_sha,
            "model_state_sha256": state["state_sha256"],
            "recipe_sha256": freeze["recipe_sha256"],
            "prefit_freeze_signature": signature,
            "fit_rows": 50608,
            "learned_unit_total": 6,
            "test_features_opened": False,
            "test_labels_opened": False,
            "sample_submission_opened": False,
        }
        trained_freeze = {**trained_body, "freeze_signature": canonical_sha(trained_body)}
        _write_json_once(TRAINED_FREEZE_PATH, trained_freeze)
        append_event({"event": "FIT_SAVED", "fit_id": FIT_ID,
                      "units": 6, "predictor_fits": 1, "preprocessing_fits": 5,
                      "model_sha256": model_sha, "receipt_sha256": file_sha(FIT_RECEIPT_PATH),
                      "trained_freeze_signature": trained_freeze["freeze_signature"]})
        return receipt
    except BaseException as exc:
        events = _read_events()
        if not any(event.get("event") in {"FIT_SAVED", "FIT_FAILED"} for event in events):
            append_event({"event": "FIT_FAILED", "fit_id": FIT_ID, "units_reserved": MAX_UNITS,
                          "failure_type": type(exc).__name__, "failure_message": str(exc)[:1000],
                          "retry_count": 0, "test_features_opened": False, "kaggle_contacted": False})
        raise


def _verify_committed_trained_state() -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(FIT_RECEIPT_PATH)
    trained = _read_json(TRAINED_FREEZE_PATH)
    body = dict(trained)
    signature = body.pop("freeze_signature", None)
    if signature != canonical_sha(body):
        raise BenchmarkError("Trained-state freeze signature mismatch")
    receipt_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha(receipt_body):
        raise BenchmarkError("C10 fit receipt signature mismatch")
    if trained.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_FEATURES":
        raise BenchmarkError("C10 trained-state freeze is not ready for inference")
    if trained.get("fit_receipt_sha256") != file_sha(FIT_RECEIPT_PATH):
        raise BenchmarkError("C10 trained-state freeze receipt hash mismatch")
    if file_sha(MODEL_PATH) != trained.get("model_file_sha256"):
        raise BenchmarkError("C10 fitted model artifact hash mismatch")
    if _git_file_commit(TRAINED_FREEZE_PATH) != _git("rev-parse", "HEAD"):
        raise BenchmarkError("Commit trained-state freeze before reading Test features")
    return receipt, trained


def _load_sparse25_inference_arrays() -> tuple[np.ndarray, ...]:
    manifest = _read_json(SPARSE25_MANIFEST)
    if file_sha(SPARSE25_MANIFEST) != EXPECTED_BASELINE_MANIFEST_SHA or file_sha(SPARSE25_BUNDLE) != EXPECTED_BASELINE_BUNDLE_SHA:
        raise BenchmarkError("Sparse25 saved inference lineage changed after freeze")
    with np.load(SPARSE25_BUNDLE, allow_pickle=False) as cached:
        required = {"prediction", "probability", "base", "missing", "sparse", "missing_mask"}
        if set(cached.files) != required:
            raise BenchmarkError("Sparse25 E79 inference array schema changed")
        arrays = tuple(np.asarray(cached[name]).copy() for name in
                       ("base", "missing", "sparse", "missing_mask", "probability", "prediction"))
    base, missing, sparse_raw, missing_mask, probability, prediction = arrays
    components = _fit_component_map(manifest)
    expected_hashes = {
        "base": components["C2_B"]["prediction_sha256"],
        "missing": components["CB_MISSING"]["prediction_sha256"],
        "sparse": components["SPARSE_BINARY"]["prediction_sha256"],
        "probability": manifest["probability_sha256"],
        "prediction": manifest["prediction_sha256"],
    }
    actual_hashes = {"base": array_digest(base), "missing": array_digest(missing),
                     "sparse": array_digest(sparse_raw), "probability": array_digest(probability),
                     "prediction": array_digest(prediction)}
    if actual_hashes != expected_hashes:
        raise BenchmarkError("Sparse25 C2_B/CatBoost/SparseBinary Test probability hashes do not match manifest")
    return base.astype(np.float64), missing.astype(np.float64), sparse_raw.astype(np.float64), \
        missing_mask.astype(bool), probability.astype(np.float64), prediction.astype(np.int64)


def infer_test() -> dict[str, Any]:
    receipt, trained = _verify_committed_trained_state()
    if any(path.exists() for path in (CSV_PATH, PREDICTIONS_PATH, INFERENCE_BUILD_PATH)):
        raise BenchmarkError("C10 E79 inference output already exists; no overwrite")
    baseline_manifest = _read_json(SPARSE25_MANIFEST)
    test_sha_expected = baseline_manifest.get("input_sha256", {}).get("test")
    if not TEST_PATH.is_file() or file_sha(TEST_PATH) != test_sha_expected:
        raise BenchmarkError("Official Test file hash does not match the frozen Sparse25 input manifest")
    test = pd.read_csv(TEST_PATH)
    if len(test) != 29421 or "artifact_signal" in test.columns:
        raise BenchmarkError("Official Test row count/schema failed; no target may be present")
    if "record_id" not in test or test["record_id"].isna().any() or test["record_id"].duplicated().any():
        raise BenchmarkError("Official Test IDs are missing or duplicated")
    ids = test["record_id"].astype(str).to_numpy()
    id_sha = hashlib.sha256(("\n".join(ids.tolist()) + "\n").encode("utf-8")).hexdigest()
    if id_sha != baseline_manifest.get("ordered_test_ids_sha256"):
        raise BenchmarkError("Official Test ID sequence differs from the accepted Sparse25 lineage")

    base, cb_missing, sparse_raw, cached_missing, sparse25_probability, sparse25_prediction = _load_sparse25_inference_arrays()
    if any(value.shape != (len(test),) for value in (base, cb_missing, sparse_raw, cached_missing,
                                                      sparse25_probability, sparse25_prediction)):
        raise BenchmarkError("Sparse25 full-Train inference arrays do not match official Test row count")
    derived_missing = np.asarray(stream_missing_mask(test), dtype=bool)
    if not np.array_equal(cached_missing, derived_missing):
        raise BenchmarkError("Sparse25 cached availability mask differs from official Test streams")
    sparse25_rebuilt = np.where(cached_missing, cb_missing, C2_WEIGHT * base + CANDIDATE_WEIGHT * sparse_raw)
    if not np.array_equal(sparse25_rebuilt, sparse25_probability):
        raise BenchmarkError("Sparse25 current full-Train probability route does not reconstruct exactly")
    if not np.array_equal((sparse25_probability >= THRESHOLD).astype(np.int64), sparse25_prediction):
        raise BenchmarkError("Sparse25 baseline labels do not reconstruct at frozen threshold")

    model = joblib.load(MODEL_PATH)
    x = _select_features(test.loc[:, list(FEATURE_NAMES)])
    with threadpool_limits(limits=2):
        raw_c10 = model.predict_proba(x)
    raw_c10 = np.asarray(raw_c10, dtype=np.float64)
    if raw_c10.shape != (len(test),) or not np.isfinite(raw_c10).all() or np.any((raw_c10 < 0) | (raw_c10 > 1)):
        raise BenchmarkError("C10 full-Train Test probabilities are invalid")
    probabilities = np.where(cached_missing, cb_missing,
                             C2_WEIGHT * base + CANDIDATE_WEIGHT * raw_c10)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise BenchmarkError("C10 routed probabilities are outside [0,1]")
    if not np.array_equal(probabilities[cached_missing], cb_missing[cached_missing]):
        raise BenchmarkError("C10 changed the frozen CatBoost missing-stream branch")
    labels = (probabilities >= THRESHOLD).astype(np.int64)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("x", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows(zip(ids.tolist(), labels.tolist()))
        output.flush()
        os.fsync(output.fileno())
    RUN.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("xb") as output:
        np.savez_compressed(output, record_id=np.asarray(ids, dtype=str),
                            raw_c10_probability=raw_c10, probabilities=probabilities,
                            labels=labels, missing_mask=cached_missing,
                            c2b_probability=base, catboost_missing_probability=cb_missing)
        output.flush()
        os.fsync(output.fileno())
    build_body = {
        "schema_version": 1,
        "status": "C10_E79_INFERENCE_BUILT_PENDING_INDEPENDENT_QA",
        "fit_id": FIT_ID,
        "trained_state_freeze_sha256": file_sha(TRAINED_FREEZE_PATH),
        "fit_receipt_sha256": file_sha(FIT_RECEIPT_PATH),
        "test_sha256": file_sha(TEST_PATH),
        "test_rows": int(len(test)),
        "ordered_test_ids_sha256": id_sha,
        "present_rows": int((~cached_missing).sum()),
        "missing_rows": int(cached_missing.sum()),
        "missing_fallback_bit_exact": True,
        "threshold": THRESHOLD,
        "predicted_label_counts": {str(int(k)): int((labels == k).sum()) for k in (0, 1)},
        "probability_sha256": array_digest(probabilities),
        "label_sha256": array_digest(labels.astype(np.int64), dtype="<i8"),
        "baseline_sparse25_labels_changed": int(np.count_nonzero(labels != sparse25_prediction)),
        "csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
        "csv_sha256": file_sha(CSV_PATH),
        "predictions_path": PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        "predictions_sha256": file_sha(PREDICTIONS_PATH),
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    build = {**build_body, "build_sha256": canonical_sha(build_body)}
    _write_json_once(INFERENCE_BUILD_PATH, build)
    return build


def _synthetic_frame(n: int) -> pd.DataFrame:
    documents = [
        "alpha beta gamma delta", "alpha beta epsilon zeta",
        "theta iota kappa lambda", "mu nu xi omicron",
    ]
    rows = []
    for i in range(n):
        missing = i % 13 == 0
        rows.append({
            "impact_load": float(10 + i % 37), "entry_friction": float(i % 9),
            "cycle_code": int(47 + i % 33), "scope_pressure": float(i % 21),
            "text_load": int(i % 17), "scope_trace": float(i % 5),
            "org_spread": float(1 + i % 4), "module_spread": float(1 + i % 6),
            "telemetry_flag": int(i % 2), "org_code": f"org-{i % 4}",
            "module_code": f"mod-{i % 5}", "pattern_code": f"pat-{i % 3}",
            "signal_stream": "" if missing else documents[(i // 3) % len(documents)],
        })
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES))


def run_synthetic_qa() -> dict[str, Any]:
    x_train = _synthetic_frame(96)
    y_train = (np.arange(len(x_train)) % 2).astype(np.int8)
    x_query = _synthetic_frame(18).copy()
    x_query.loc[0, "signal_stream"] = ""
    x_query.loc[1, "signal_stream"] = "omega sigma tau upsilon"
    model = SeparateTrigramSparseClassifier().fit(x_train, y_train)
    p_direct = model.predict_proba(x_query)
    with tempfile.TemporaryDirectory(prefix="c10-e79-synthetic-") as temp:
        model_path = Path(temp) / "model.joblib"
        joblib.dump(model, model_path)
        resumed = joblib.load(model_path)
        p_resumed = resumed.predict_proba(x_query)
    if not np.array_equal(p_direct, p_resumed):
        raise BenchmarkError("Synthetic persistence/resume predictions differ")
    if model.audit_state()["fit_rows"] != len(x_train):
        raise BenchmarkError("Synthetic model fit population is not accounted")
    base = np.linspace(0.1, 0.9, len(p_direct), dtype=np.float64)
    cb = np.linspace(0.8, 0.2, len(p_direct), dtype=np.float64)
    missing = np.arange(len(p_direct)) % 4 == 0
    routed = np.where(missing, cb, C2_WEIGHT * base + CANDIDATE_WEIGHT * p_direct)
    if not np.array_equal(routed[missing], cb[missing]):
        raise BenchmarkError("Synthetic C10 missing-stream fallback is not bit-exact")
    if len(model.trigram.vocabulary_) == 0 or not np.isfinite(routed).all():
        raise BenchmarkError("Synthetic C10 trigram or routing QA failed")
    qa = {
        "schema_version": 1,
        "status": "PASS_SYNTHETIC_C10_END_TO_END_AND_RESUME",
        "synthetic_rows": {"fit": len(x_train), "query": len(x_query)},
        "synthetic_predictor_fits": 1,
        "synthetic_preprocessing_units": 5,
        "official_predictor_fits": 0,
        "trigram_features": int(len(model.trigram.vocabulary_)),
        "persistence_resume_bit_exact": True,
        "missing_route_bit_exact": True,
        "invalid_probability_count": int(np.count_nonzero(~np.isfinite(routed) | (routed < 0) | (routed > 1))),
        "test_features_opened": False,
        "test_labels_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--write-prefit-freeze", action="store_true")
    group.add_argument("--arm-fit", action="store_true")
    group.add_argument("--fit-full-train", action="store_true")
    group.add_argument("--infer-test", action="store_true")
    args = parser.parse_args()
    if args.synthetic_qa:
        result = run_synthetic_qa()
        _write_json_once(SYNTHETIC_QA_PATH, result)
    elif args.write_prefit_freeze:
        result = write_prefit_freeze()
    elif args.arm_fit:
        result = arm_fit()
    elif args.fit_full_train:
        result = fit_full_train()
    else:
        result = infer_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
