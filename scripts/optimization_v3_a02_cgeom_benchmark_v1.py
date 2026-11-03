"""Prepare three frozen full-Train Classification benchmark CSVs, without upload.

The runner uses a separate namespace and the append-only Research Next budget.
Each learned fit is durably accounted for before fitting and persisted once.
Test features are opened only after ``freeze-trained`` creates the trained-state
freeze; Test labels are excluded from every reader and are never loaded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_a02_continuation as a02_runner  # noqa: E402
from scripts import optimization_v3_c19_benchmark as c19  # noqa: E402
from scripts import optimization_v3_positional_text_pilot_v1 as positional  # noqa: E402
from src.models import classification_a02_neighbors as a02  # noqa: E402
from src.models.a02_coefficient_geometry_v1 import CoefficientGeometryPredictor  # noqa: E402
from src.models.classification_models import EXPANDED_FEATURES, _TFIDF_PARAMS  # noqa: E402
from src.models.research_next_budget import read_events  # noqa: E402


BATCH_ID = "A02_CGEOM_BENCHMARK_V1"
CARD_ID = BATCH_ID
STAGE = "THREE_CSV_E79_FULLTRAIN"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
AMENDMENT_ID = "A02_CGEOM_BENCHMARK_CAP130_2026_11_03"
AUTHORIZED_BASE_HEAD = "f3baa4014378a9fcb1a58e898d1b4c9a87679728"
INITIAL_PREPARATION_COMMIT = "cc8e406a13fdea09c01b0f2356d1b7c8b822a470"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_TEST_SHA = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
EXPECTED_SAMPLE_SHA = "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7"
INITIAL_LEDGER_SHA = "5bd8aa218e17268913de9ecdac5b85ea576e611cde5566f2b3d242f371ff317a"
EXPECTED_LEDGER_ROWS = 303
EXPECTED_STARTS = 118
EXPECTED_SAVED = 114
EXPECTED_FAILED = 4
BASE_CAP = 121
AMENDED_CAP = 130
BASE_CATEGORY_CAPS = {"pilot": 80, "continuation": 38, "control_replay": 2, "technical_retry": 1}
AMENDED_CATEGORY_CAPS = {"pilot": 80, "continuation": 47, "control_replay": 2, "technical_retry": 1}
MAX_NEW_FITS = 9
THREAD_LIMIT = 2
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
THRESHOLD = 0.50

TRAIN = ROOT / "data/raw/classification/Train.csv"
TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1"
UNIT_ROOT = RUN_ROOT / "units"
OUTPUT_NPZ = RUN_ROOT / "test_predictions.npz"
PREFLIGHT_PATH = RUN_ROOT / "PREFLIGHT.json"
SYNTHETIC_QA = RUN_ROOT / "SYNTHETIC_QA.json"
TRAINED_FREEZE = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1/E79_TRAINED_FREEZE.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1"
RECIPE_PATH = REPORT_ROOT / "FROZEN_RECIPES.json"
BATCH_AMENDMENT_PATH = REPORT_ROOT / "BATCH_AMENDMENT.md"
MANIFEST_PATH = REPORT_ROOT / "BATCH_MANIFEST.json"
REPORT_PATH = REPORT_ROOT / "BATCH_REPORT.md"
OUTPUT_ROOT = ROOT / "submissions/optimization_v3/a02_cgeom_benchmark_v1"

A02_ROOT = ROOT / "experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed"
A02_MODEL = A02_ROOT / "model.joblib"
A02_STATE = A02_ROOT / "fit_state.json"
A02_FIT_RECEIPT = A02_ROOT / "fit_receipt.json"
A02_TEST_NPZ = A02_ROOT / "test_predictions.npz"
A02_TEST_RECEIPT = A02_ROOT / "test_predictions.json"
A02_MANIFEST = ROOT / "reports/optimization_v3/a02_continuation_v1/submission_manifest.json"
SPARSE25_CACHE = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"
SPARSE25_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
SPARSE25_CSV = ROOT / "submissions/dual_campaign/optimization_v3/v3_cls_sparse25.csv"
C16_DIR = ROOT / "experiments/runs/c16_r22_submission"
C16_AUDIT = C16_DIR / "classification_c16_fit_audit.json"
C16_OUTPUTS = C16_DIR / "classification_c16_outputs.npz"

FIT_IDS = (
    f"{BATCH_ID}_E79_C2B_STRUCTURED",
    f"{BATCH_ID}_E79_C2B_SCALER",
    f"{BATCH_ID}_E79_C2B_OHE",
    f"{BATCH_ID}_E79_C2B_LR",
    f"{BATCH_ID}_E79_C1B_STRUCTURED",
    f"{BATCH_ID}_E79_C1B_HGB",
    f"{BATCH_ID}_E79_POS_BAG_OFFSET_GAMMA",
    f"{BATCH_ID}_E79_JOINTPOS_LR",
    f"{BATCH_ID}_E79_CGEOM_ALIGNED_PREDICTOR",
)
UNIT_COMPONENTS = (
    "C2_B_STRUCTURED_PREPROCESSOR",
    "C2_B_NUMERIC_SCALER",
    "C2_B_ONE_HOT_ENCODER",
    "C2_B_LOGISTIC_REGRESSION",
    "C1_B_STRUCTURED_PREPROCESSOR",
    "C1_B_HIST_GRADIENT_BOOSTING",
    "POS_BAG_OFFSET_LBFGS_GAMMA",
    "JOINT_POSITION_LOGISTIC_REGRESSION",
    "CGEOM_ALIGNED_MEMORY_PREDICTOR",
)
SOURCE_PATHS = (
    "scripts/optimization_v3_a02_cgeom_benchmark_v1.py",
    "tests/test_optimization_v3_a02_cgeom_benchmark_v1.py",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/BATCH_AMENDMENT.md",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/FROZEN_RECIPES.json",
    "src/models/classification_models.py",
    "src/models/round1_models.py",
    "src/models/classification_a02_neighbors.py",
    "src/models/a02_coefficient_geometry_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "src/models/research_next_budget.py",
    "src/models/positional_bag_offset_v1.py",
    "scripts/optimization_v3_a02_coefficient_geometry_v1.py",
    "scripts/optimization_v3_a02_continuation.py",
    "scripts/optimization_v3_a02_final_inference_audit.py",
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/optimization_v3_c19_benchmark.py",
    "reports/optimization_v3/a02_coefficient_geometry_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_pilot_v1/RECIPE.json",
    "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
)
CLASSIFICATION_FEATURES = tuple(EXPANDED_FEATURES) + ("signal_stream",)


class BatchError(RuntimeError):
    """Frozen batch preparation failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise BatchError(f"Refusing to replace immutable artifact: {path}")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_once(path: Path, value: Any) -> None:
    write_once(path, json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                                 allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8") + b"\n")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def verify_preparation_head() -> str:
    """Allow the frozen prep commit or one direct, scoped QA-repair child."""
    head = _git("rev-parse", "HEAD")
    parents = _git("rev-list", "--parents", "-n", "1", head).split()[1:]
    if head == INITIAL_PREPARATION_COMMIT:
        if parents != [AUTHORIZED_BASE_HEAD]:
            raise BatchError("Initial preparation commit is not the authorized direct child")
    elif parents != [INITIAL_PREPARATION_COMMIT]:
        raise BatchError("Source must be the frozen preparation commit or its one direct QA-repair child")
    return head


def require_clean_tree() -> None:
    if _git("status", "--porcelain", "--untracked-files=normal"):
        raise BatchError("Tracked worktree must be clean at a freeze boundary")


def source_hashes() -> dict[str, str]:
    missing = [item for item in SOURCE_PATHS if not (ROOT / item).is_file()]
    if missing:
        raise BatchError(f"Required recipe/source files are missing: {missing}")
    return {item: sha256_file(ROOT / item) for item in SOURCE_PATHS}


def input_hashes() -> dict[str, str]:
    paths = {
        "data/raw/classification/Train.csv": (TRAIN, EXPECTED_TRAIN_SHA),
        "data/raw/classification/Test.csv": (TEST, EXPECTED_TEST_SHA),
        "data/raw/classification/SampleSubmission.csv": (SAMPLE, EXPECTED_SAMPLE_SHA),
    }
    actual: dict[str, str] = {}
    for name, (path, expected) in paths.items():
        if not path.is_file():
            raise BatchError(f"Required official input is missing: {name}")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise BatchError(f"Official input hash mismatch: {name}")
    return actual


def _budget_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    caps = None
    cap = None
    for event in events:
        if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED":
            cap = int(event["cap"])
            caps = dict(event["category_caps"])
        elif event.get("event") == "BUDGET_CATEGORY_AMENDMENT":
            caps = dict(event["effective_caps"])
        elif event.get("event") == "BUDGET_CAP_AMENDMENT":
            cap = int(event["global_cap"])
            caps = dict(event["effective_caps"])
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    releases = {event.get("reservation_key"): int(event.get("unused_fits", 0))
                for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active_by_category: dict[str, int] = {}
    for event in reservations:
        active = int(event["fits"]) - releases.get(event.get("reservation_key"), 0)
        category = str(event.get("category"))
        active_by_category[category] = active_by_category.get(category, 0) + active
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    started_ids = {event.get("fit_id") for event in starts}
    terminal_ids = {event.get("fit_id") for event in (*saved, *failed)}
    return {
        "event_rows": len(events), "global_cap": cap, "category_caps": caps,
        "active_reservations_by_category": active_by_category,
        "active_reservations": sum(active_by_category.values()),
        "fit_starts": len(starts), "fit_saved": len(saved), "fit_failed": len(failed),
        "fit_pending": len(started_ids - terminal_ids),
    }


def verify_initial_budget() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not LEDGER.is_file() or sha256_file(LEDGER) != INITIAL_LEDGER_SHA:
        raise BatchError("Shared Research Next ledger differs from the verified pre-batch checkpoint")
    events = read_events(LEDGER)
    snapshot = _budget_snapshot(events)
    expected = {
        "event_rows": EXPECTED_LEDGER_ROWS, "global_cap": BASE_CAP,
        "category_caps": BASE_CATEGORY_CAPS,
        "active_reservations_by_category": {"pilot": 77, "continuation": 38, "control_replay": 2, "technical_retry": 1},
        "active_reservations": 118, "fit_starts": EXPECTED_STARTS,
        "fit_saved": EXPECTED_SAVED, "fit_failed": EXPECTED_FAILED, "fit_pending": 0,
    }
    if snapshot != expected:
        raise BatchError(f"Research Next budget does not match the approved checkpoint: {snapshot}")
    own_events = [event for event in events if event.get("card_id") == CARD_ID or event.get("reservation_key") == RESERVATION_KEY]
    if own_events:
        raise BatchError("New batch events already exist before its amendment/reservation")
    return events, snapshot


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchError(f"Expected JSON object at {path}")
    return value


def _verify_reuse_artifacts() -> dict[str, Any]:
    required = (A02_MODEL, A02_STATE, A02_FIT_RECEIPT, A02_TEST_NPZ, A02_TEST_RECEIPT,
                A02_MANIFEST, SPARSE25_CACHE, SPARSE25_MANIFEST, SPARSE25_CSV,
                C16_AUDIT, C16_OUTPUTS)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise BatchError(f"A02/C16 lineage artifacts missing: {missing}")
    manifest = _read_json(A02_MANIFEST)
    a02_fit_receipt = _read_json(A02_FIT_RECEIPT)
    a02_test_receipt = _read_json(A02_TEST_RECEIPT)
    c16_audit = _read_json(C16_AUDIT)
    a02_freeze = a02_runner.verify_freeze()
    if (manifest.get("model_sha256") != sha256_file(A02_MODEL)
            or a02_fit_receipt.get("fit_id") != "A02_CONT_E79_OBSERVED_FULLTRAIN"):
        raise BatchError("A02 E79 model or receipt does not match its manifest")
    receipt_body = dict(a02_fit_receipt)
    receipt_sha = receipt_body.pop("receipt_sha256", None)
    if receipt_sha != canonical_sha(receipt_body) or receipt_sha != manifest.get("fit_receipt_sha256"):
        raise BatchError("A02 E79 fit receipt hash does not verify")
    if (manifest.get("prediction_bundle_sha256") != sha256_file(A02_TEST_NPZ)
            or a02_test_receipt.get("npz_sha256") != sha256_file(A02_TEST_NPZ)
            or a02_test_receipt.get("fit_receipt_sha256") != receipt_sha
            or a02_test_receipt.get("freeze_signature") != a02_fit_receipt.get("freeze_signature")
            or a02_test_receipt.get("official_test_targets_loaded") is not False
            or manifest.get("test_target_loaded") is not False
            or manifest.get("test_feature_sha256") != EXPECTED_TEST_SHA
            or manifest.get("sample_submission_sha256") != EXPECTED_SAMPLE_SHA):
        raise BatchError("A02 saved Test prediction receipt/manifest lineage does not verify")
    fit_state = _read_json(A02_STATE)
    if (a02_fit_receipt.get("model_sha256") != sha256_file(A02_MODEL)
            or a02_fit_receipt.get("fit_rows") not in (None, 50_608)
            or a02_fit_receipt.get("text_prefix_rows") != 50_608
            or a02_fit_receipt.get("test_data_used") is not False
            or a02_fit_receipt.get("query_labels_passed_to_fit") is not False
            or manifest.get("official_train_sha256") != EXPECTED_TRAIN_SHA
            or manifest.get("official_train_rows") != 50_608):
        raise BatchError("A02 E79 fit receipt/manifest has an invalid population or data boundary")
    model = joblib.load(A02_MODEL)
    if model.audit_state() != fit_state or model.text_space.fit_rows != 50_608 or model.text_space.origin_cycle != 79:
        raise BatchError("A02 E79 persisted model does not match its fit-state receipt")
    with np.load(C16_OUTPUTS, allow_pickle=False) as archive:
        c16_keys = set(archive.files)
        if any(np.asarray(archive[name]).dtype.kind == "O" for name in archive.files):
            raise BatchError("C16 output cache contains object dtype")
    c16_reused = c16_audit.get("reused_full_train_components", {}).get("reused_models", {})
    if (c16_audit.get("task") != "classification" or c16_audit.get("candidate") != "C16"
            or c16_reused.get("C2_B", {}).get("fit_rows") != 50_608
            or c16_reused.get("CB_MISSING", {}).get("fit_rows") != 50_608
            or not c16_keys.issuperset({"base", "missing", "stream_missing"})
            or any(path.suffix == ".joblib" for path in C16_DIR.glob("*"))):
        raise BatchError("C16 cache audit does not support the stated prediction-only reuse finding")
    return {
        "a02_model_sha256": sha256_file(A02_MODEL),
        "a02_freeze_signature": a02_freeze.get("freeze_signature"),
        "a02_fit_state_sha256": canonical_sha(fit_state),
        "a02_fit_receipt_sha256": sha256_file(A02_FIT_RECEIPT),
        "a02_test_predictions_sha256": sha256_file(A02_TEST_NPZ),
        "a02_test_receipt_sha256": sha256_file(A02_TEST_RECEIPT),
        "a02_manifest_sha256": sha256_file(A02_MANIFEST),
        "sparse25_cache_sha256": sha256_file(SPARSE25_CACHE),
        "sparse25_manifest_sha256": sha256_file(SPARSE25_MANIFEST),
        "sparse25_csv_sha256": sha256_file(SPARSE25_CSV),
        "c16_prediction_only_audit_sha256": sha256_file(C16_AUDIT),
        "c16_prediction_only_bundle_sha256": sha256_file(C16_OUTPUTS),
        "c16_source_commit": c16_audit.get("reused_full_train_components", {}).get("source_commit"),
        "c16_reuse_cache_sha256": c16_audit.get("reused_full_train_components", {}).get("cache_sha256"),
        "c16_reuse_manifest_sha256": c16_audit.get("reused_full_train_components", {}).get("manifest_sha256"),
        "c16_cached_output_keys": sorted(c16_keys),
        "c16_cached_estimators_reusable": False,
        "c2b_fulltrain_estimator_reusable": False,
        "c1b_fulltrain_estimator_reusable": False,
        "a02_model_state_valid": True,
    }


def preflight() -> dict[str, Any]:
    require_clean_tree()
    head = verify_preparation_head()
    if json.loads(RECIPE_PATH.read_text(encoding="utf-8")).get("authorized_base_head") != AUTHORIZED_BASE_HEAD:
        raise BatchError("Recipe manifest starting HEAD does not match user instruction")
    if not BATCH_AMENDMENT_PATH.is_file():
        raise BatchError("Batch amendment document is missing")
    events, budget = verify_initial_budget()
    inputs = input_hashes()
    reuse = _verify_reuse_artifacts()
    running = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (process.info.get("name") or "").lower()
            command = " ".join(process.info.get("cmdline") or []).lower()
            relevant = any(token in command for token in ("pytest", "optimization_v3", "train", "fit"))
            if process.info.get("pid") != os.getpid() and "python" in name and relevant:
                running.append({"pid": process.info["pid"], "command": command[:240]})
        except psutil.Error:
            continue
    if running:
        raise BatchError(f"Another batch process is active: {running}")
    vm = psutil.virtual_memory()
    if vm.available < INITIAL_FREE_MIN:
        raise BatchError(f"Initial RAM free is below 4 GiB: {vm.available}")
    if (RUN_ROOT / ".training.lock").exists():
        raise BatchError("Batch training lock already exists; inspect owner before resume")
    if UNIT_ROOT.exists() and any(UNIT_ROOT.iterdir()):
        raise BatchError("Unexpected partial fitted-unit inventory exists before official fitting")
    result = {
        "status": "PASS_PREFIT_AND_REUSE_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_base_head": AUTHORIZED_BASE_HEAD,
        "preparation_head": head,
        "preparation_commit_is_direct_child": True,
        "working_tree_clean": True,
        "ledger": {"sha256": sha256_file(LEDGER), **budget},
        "official_input_sha256": inputs,
        "existing_model_reuse_audit": reuse,
        "fit_units": list(FIT_IDS), "new_fit_maximum": MAX_NEW_FITS,
        "available_ram_bytes": int(vm.available), "training_processes_for_this_batch": running,
        "legacy_batch1_lock_present_preserved": (ROOT / "experiments/runs/optimization_v3_batch1/.training.lock").exists(),
        "test_features_parsed": False, "sample_submission_parsed": False,
        "test_targets_loaded": False, "kaggle_contacted": False,
    }
    if PREFLIGHT_PATH.exists():
        previous = _read_json(PREFLIGHT_PATH)
        for key in ("authorized_base_head", "preparation_head", "ledger", "official_input_sha256",
                    "existing_model_reuse_audit", "fit_units", "new_fit_maximum"):
            if previous.get(key) != result.get(key):
                raise BatchError("Existing preflight receipt differs from current audited state")
        return previous
    write_json_once(PREFLIGHT_PATH, result)
    return result


def _append_event(event: Mapping[str, Any], *, ledger_path: Path, synthetic: bool = False) -> dict[str, Any]:
    resolved = ledger_path.resolve()
    if synthetic and resolved == LEDGER.resolve():
        raise BatchError("Synthetic mode is prohibited from writing to the shared Research Next ledger")
    if not synthetic and resolved != LEDGER.resolve():
        raise BatchError("Official fit accounting accepts only the configured shared ledger")
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = canonical_bytes(payload) + b"\n"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ledger_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return payload


def _verify_batch_amendment(events: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    amendments = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT" and event.get("amendment_id") == AMENDMENT_ID]
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED" and event.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) > 1 or len(reservations) > 1:
        raise BatchError("Duplicate amendment or reservation events found")
    return (amendments[0] if amendments else None, reservations[0] if reservations else None)


def amend_and_reserve() -> dict[str, Any]:
    events = read_events(LEDGER)
    amendment, reservation = _verify_batch_amendment(events)
    if amendment is None:
        snapshot = _budget_snapshot(events)
        if (sha256_file(LEDGER) != INITIAL_LEDGER_SHA or snapshot["event_rows"] != EXPECTED_LEDGER_ROWS
                or snapshot["global_cap"] != BASE_CAP or snapshot["category_caps"] != BASE_CATEGORY_CAPS
                or snapshot["active_reservations"] != 118 or snapshot["fit_starts"] != 118
                or snapshot["fit_saved"] != 114 or snapshot["fit_failed"] != 4 or snapshot["fit_pending"] != 0):
            raise BatchError("Budget changed after pre-fit audit; refusing cap amendment")
        payload = {
            "event": "BUDGET_CAP_AMENDMENT", "amendment_id": AMENDMENT_ID,
            "authorized_by": "Sol-steered Samuel batch authorization; exactly nine maximum fit units",
            "batch_id": "optimization_v3_research_next_v1", "card_id": CARD_ID,
            "prior_global_cap": BASE_CAP, "global_cap": AMENDED_CAP,
            "prior_effective_caps": BASE_CATEGORY_CAPS, "effective_caps": AMENDED_CATEGORY_CAPS,
            "active_reservations_before": {"pilot": 77, "continuation": 38, "control_replay": 2, "technical_retry": 1},
            "active_total_before": 118,
            "historical_started_saved_failed_before": {"started": 118, "saved": 114, "failed": 4, "pending": 0},
            "pre_amendment_event_rows": EXPECTED_LEDGER_ROWS,
            "pre_amendment_ledger_sha256": INITIAL_LEDGER_SHA,
            "continuation_capacity_added": 9,
            "other_category_caps_unchanged": True,
            "maximum_new_fit_units": MAX_NEW_FITS,
            "maximum_total_starts": 127,
            "reason": "Authorized three-candidate E79 Classification batch: shared C2_B and C1_B state plus POS_BAG_OFFSET, JOINTPOS, and exact aligned CGEOM; no retries, other lanes, or upload.",
            "source_commit": _git("rev-parse", "HEAD"),
        }
        _append_event(payload, ledger_path=LEDGER)
        events = read_events(LEDGER)
        amendment, reservation = _verify_batch_amendment(events)
    if amendment is None:
        raise BatchError("Cap amendment was not durably recorded")
    expected_amendment = {
        "prior_global_cap": BASE_CAP, "global_cap": AMENDED_CAP,
        "prior_effective_caps": BASE_CATEGORY_CAPS, "effective_caps": AMENDED_CATEGORY_CAPS,
        "pre_amendment_event_rows": EXPECTED_LEDGER_ROWS,
        "pre_amendment_ledger_sha256": INITIAL_LEDGER_SHA,
        "maximum_new_fit_units": MAX_NEW_FITS, "maximum_total_starts": 127,
    }
    if any(amendment.get(key) != value for key, value in expected_amendment.items()):
        raise BatchError("Existing cap amendment differs from the authorized scope")
    if reservation is None:
        snapshot = _budget_snapshot(events)
        if (snapshot["global_cap"] != AMENDED_CAP or snapshot["category_caps"] != AMENDED_CATEGORY_CAPS
                or snapshot["active_reservations"] != 118
                or snapshot["active_reservations_by_category"].get("continuation") != 38
                or snapshot["fit_starts"] != 118):
            raise BatchError("Reservation preconditions changed after cap amendment")
        _append_event({
            "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "category": "continuation", "fits": MAX_NEW_FITS, "fit_ids": list(FIT_IDS),
            "global_cap": AMENDED_CAP, "effective_category_cap": AMENDED_CATEGORY_CAPS["continuation"],
            "amendment_id": AMENDMENT_ID, "source_commit": _git("rev-parse", "HEAD"),
        }, ledger_path=LEDGER)
        events = read_events(LEDGER)
        amendment, reservation = _verify_batch_amendment(events)
    if (reservation is None or reservation.get("fits") != MAX_NEW_FITS
            or reservation.get("fit_ids") != list(FIT_IDS)
            or reservation.get("category") != "continuation"
            or reservation.get("global_cap") != AMENDED_CAP):
        raise BatchError("Nine-unit reservation does not match frozen fit IDs")
    snapshot = _budget_snapshot(events)
    if (snapshot["active_reservations"] != 127 or snapshot["fit_starts"] != 118
            or snapshot["category_caps"] != AMENDED_CATEGORY_CAPS
            or snapshot["active_reservations_by_category"].get("continuation") != 47):
        raise BatchError(f"Post-reservation shared budget state is invalid: {snapshot}")
    return {"status": "PASS_CAP_AMENDED_AND_NINE_IDS_RESERVED", "ledger_sha256": sha256_file(LEDGER), **snapshot}


def _unit_path(fit_id: str) -> Path:
    if fit_id not in FIT_IDS:
        raise BatchError(f"Unexpected fit ID: {fit_id}")
    return UNIT_ROOT / f"{fit_id}.joblib"


def _unit_receipt_path(fit_id: str) -> Path:
    return UNIT_ROOT / f"{fit_id}.json"


def _dump_joblib_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BatchError(f"Refusing to replace fitted model artifact: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        joblib.dump(value, tmp, compress=0)
        with tmp.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_unit(fit_id: str) -> Any:
    path, receipt_path = _unit_path(fit_id), _unit_receipt_path(fit_id)
    if not path.is_file() or not receipt_path.is_file():
        raise BatchError(f"Required persisted fit state is missing: {fit_id}")
    receipt = _read_json(receipt_path)
    if receipt.get("fit_id") != fit_id or receipt.get("artifact_sha256") != sha256_file(path):
        raise BatchError(f"Persisted state receipt mismatch: {fit_id}")
    return joblib.load(path)


def _fit_events(events: list[dict[str, Any]], fit_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminals = [event for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and event.get("fit_id") == fit_id]
    if len(starts) > 1 or len(terminals) > 1:
        raise BatchError(f"Duplicate budget accounting for fit {fit_id}")
    return starts, terminals


def _verify_receipt(fit_id: str, expected_head: str | None = None,
                    artifact_root: Path | None = None) -> dict[str, Any]:
    root = artifact_root if artifact_root is not None else UNIT_ROOT
    path = root / f"{fit_id}.joblib"
    receipt_path = root / f"{fit_id}.json"
    if not path.is_file() or not receipt_path.is_file():
        raise BatchError(f"Durable fitted artifact/receipt incomplete for {fit_id}")
    receipt = _read_json(receipt_path)
    body = dict(receipt)
    receipt_hash = body.pop("receipt_sha256", None)
    synthetic = receipt.get("synthetic") is True
    if (receipt.get("fit_id") != fit_id or receipt.get("batch_id") != BATCH_ID
            or receipt.get("fit_population_rows") != (8 if synthetic else 50_608)
            or receipt.get("train_sha256") != ("synthetic" if synthetic else EXPECTED_TRAIN_SHA)
            or receipt.get("test_data_used_for_fit") is not False
            or receipt.get("validation_or_test_labels_passed_to_fit") is not False
            or receipt.get("artifact_sha256") != sha256_file(path)
            or receipt_hash != canonical_sha(body)
            or (expected_head is not None and receipt.get("preparation_commit") != expected_head)):
        raise BatchError(f"Fitted state receipt fails immutable lineage verification: {fit_id}")
    return receipt


class PeakRSS:
    def __init__(self) -> None:
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSS":
        process = psutil.Process()

        def monitor() -> None:
            while not self.stop.wait(0.1):
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except psutil.Error:
                    return

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.peak = max(self.peak, psutil.Process().memory_info().rss)


@contextmanager
def training_lock():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = RUN_ROOT / ".training.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BatchError("Batch lock exists; no process will remove or override it") from exc
    owner = json.dumps({"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()}).encode("utf-8")
    try:
        os.write(descriptor, owner)
        os.fsync(descriptor)
        os.close(descriptor)
        yield
    finally:
        try:
            if lock.read_bytes() == owner:
                lock.unlink()
        except FileNotFoundError:
            pass


def _resource_snapshot(*, first_fit: bool, synthetic: bool = False) -> dict[str, int]:
    vm = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    concurrent = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (process.info.get("name") or "").lower()
            command = " ".join(process.info.get("cmdline") or []).lower()
            if (process.info.get("pid") != os.getpid() and "python" in name
                    and any(token in command for token in ("optimization_v3", "pytest", "train", "fit"))):
                concurrent.append(process.info["pid"])
        except psutil.Error:
            continue
    if concurrent and not synthetic:
        raise BatchError(f"Another Python training/research process is active: {concurrent}")
    if vm.available < FREE_RESERVE or rss > RSS_LIMIT:
        raise BatchError(f"Resource gate failed before fit: available={vm.available}, rss={rss}")
    if first_fit and vm.available < INITIAL_FREE_MIN:
        raise BatchError(f"First-fit RAM free below 4 GiB: {vm.available}")
    return {"available_ram_bytes": int(vm.available), "rss_before_bytes": int(rss),
            "free_reserve_bytes": FREE_RESERVE, "rss_ceiling_bytes": RSS_LIMIT,
            "threads": THREAD_LIMIT, "concurrent_jobs": 1,
            "other_training_processes": [] if synthetic else concurrent,
            "synthetic_mode": synthetic}


def _unit_population(fit_id: str, *, synthetic: bool = False) -> dict[str, Any]:
    if synthetic:
        return {"rows": 8, "cycles": [47, 54], "target": "synthetic_y", "test_used": False}
    feature_only = {FIT_IDS[0], FIT_IDS[1], FIT_IDS[2], FIT_IDS[4], FIT_IDS[8]}
    detail: dict[str, Any] = {"rows": 50_608, "cycles": [47, 79],
                              "target_values_used_for_fit": fit_id not in feature_only,
                              "source": "official Classification Train"}
    if fit_id == FIT_IDS[8]:
        detail.update({"a02_text_space_rows": 50_608, "oof_residual_donor_rows": 36_010,
                       "oof_residual_donor_cycles": [56, 79],
                       "target_source": "verified full-Train C16 OOF residual bank; no new label fitting"})
    return detail


def execute_fit_unit(
    fit_id: str,
    component: str,
    callback: Callable[[], Any],
    *,
    synthetic: bool = False,
    ledger_path: Path = LEDGER,
    interrupt_after_receipt: bool = False,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise BatchError(f"Fit ID is not in the frozen nine-unit reservation: {fit_id}")
    if synthetic and ledger_path.resolve() == LEDGER.resolve():
        raise BatchError("Synthetic unit cannot write to shared production ledger")
    if not synthetic and ledger_path.resolve() != LEDGER.resolve():
        raise BatchError("Official fit unit must write to the shared production ledger")
    events = read_events(ledger_path)
    starts, terminals = _fit_events(events, fit_id)
    root = artifact_root if artifact_root is not None else UNIT_ROOT
    artifact, receipt_path = root / f"{fit_id}.joblib", root / f"{fit_id}.json"
    if terminals:
        if terminals[0].get("event") == "FIT_FAILED":
            raise BatchError(f"Fit {fit_id} already failed; retry is prohibited")
        receipt = _verify_receipt(fit_id, artifact_root=artifact_root)
        if terminals[0].get("receipt_sha256") != sha256_file(receipt_path):
            raise BatchError(f"Saved ledger receipt hash mismatch for {fit_id}")
        return {"status": "RESUMED_SAVED", "fit_id": fit_id, "receipt_sha256": sha256_file(receipt_path)}
    if starts:
        if artifact.is_file() and receipt_path.is_file():
            receipt = _verify_receipt(fit_id, artifact_root=artifact_root)
            _append_event({"event": "FIT_SAVED", "batch_id": BATCH_ID, "reservation_key": RESERVATION_KEY,
                           "card_id": CARD_ID, "stage": STAGE, "fit_id": fit_id,
                           "cutoff": 79, "component": component, "receipt_sha256": sha256_file(receipt_path),
                           "recovered_durable_receipt": True}, ledger_path=ledger_path, synthetic=synthetic)
            return {"status": "RECOVERED_DURABLE_RECEIPT", "fit_id": fit_id,
                    "receipt_sha256": sha256_file(receipt_path), "state_sha256": receipt["state_sha256"]}
        _append_event({"event": "FIT_FAILED", "batch_id": BATCH_ID, "reservation_key": RESERVATION_KEY,
                       "card_id": CARD_ID, "stage": STAGE, "fit_id": fit_id,
                       "cutoff": 79, "component": component,
                       "error": "started unit lacks complete durable state; retry prohibited"},
                      ledger_path=ledger_path, synthetic=synthetic)
        raise BatchError(f"Started unit {fit_id} has no durable receipt; recorded terminal failure, no retry")
    if artifact.exists() or receipt_path.exists():
        raise BatchError(f"Unaccounted fitted output exists before start for {fit_id}")
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED" and event.get("reservation_key") == RESERVATION_KEY]
    if len(reservation) != 1 or fit_id not in reservation[0].get("fit_ids", []):
        raise BatchError(f"No unique reservation includes fit {fit_id}")
    used = sum(event.get("event") == "FIT_STARTED" and event.get("reservation_key") == RESERVATION_KEY for event in events)
    if used >= MAX_NEW_FITS or len([e for e in events if e.get("event") == "FIT_STARTED"]) >= 127:
        raise BatchError("Authorized maximum fit starts reached")
    first_fit = not any(event.get("event") == "FIT_STARTED" and event.get("reservation_key") == RESERVATION_KEY for event in events)
    resource = _resource_snapshot(first_fit=first_fit, synthetic=synthetic)
    prep_head = _git("rev-parse", "HEAD")
    if not synthetic:
        frozen = _read_json(RECIPE_PATH)
        if frozen.get("status") != "PRE_FIT_RECIPE_FREEZE":
            raise BatchError("Pre-fit recipe freeze is missing or changed")
        src = source_hashes()
        train_hash = sha256_file(TRAIN)
        if train_hash != EXPECTED_TRAIN_SHA:
            raise BatchError("Official Train hash changed before fit")
        fit_population = _unit_population(fit_id)
    else:
        src, train_hash = {"synthetic": "synthetic"}, "synthetic"
        fit_population = _unit_population(fit_id, synthetic=True)
    start_time = datetime.now(timezone.utc).isoformat()
    _append_event({"event": "FIT_STARTED", "batch_id": BATCH_ID, "reservation_key": RESERVATION_KEY,
                   "card_id": CARD_ID, "stage": STAGE, "fit_id": fit_id, "cutoff": 79,
                   "component": component, "fit_kind": "FROZEN_FULLTRAIN_COMPONENT_OR_PREDICTOR",
                   "fit_consumed_at_start": True, "amendment_id": None if synthetic else AMENDMENT_ID,
                   "preparation_commit": prep_head, "resource_snapshot": resource,
                   "fit_population": fit_population}, ledger_path=ledger_path, synthetic=synthetic)
    start = time.perf_counter()
    try:
        with PeakRSS() as monitor, threadpool_limits(limits=THREAD_LIMIT):
            fitted = callback()
            state_sha = canonical_sha(_state_summary(fitted))
            _dump_joblib_once(artifact, fitted)
            receipt = {
                "schema_version": 1, "batch_id": BATCH_ID, "fit_id": fit_id,
                "component": component, "fit_consumed_at_start": True,
                "synthetic": bool(synthetic),
                "preparation_commit": prep_head, "source_sha256": src,
                "train_path": str(TRAIN.relative_to(ROOT)) if not synthetic else "synthetic",
                "train_sha256": train_hash, "fit_population_rows": 50_608 if not synthetic else 8,
                "fit_population": fit_population,
                "fit_cycle_min_max": [47, 79] if not synthetic else [47, 54],
                "state_sha256": state_sha, "artifact_sha256": sha256_file(artifact),
                "test_data_used_for_fit": False, "validation_or_test_labels_passed_to_fit": False,
                "resource_before": resource, "peak_rss_bytes": int(monitor.peak),
                "rss_ceiling_bytes": RSS_LIMIT, "threads": THREAD_LIMIT, "jobs": 1,
                "started_at_utc": start_time, "elapsed_seconds": float(time.perf_counter() - start),
            }
            receipt["receipt_sha256"] = canonical_sha(receipt)
            write_json_once(receipt_path, receipt)
            if monitor.peak > RSS_LIMIT:
                raise BatchError(f"Fit {fit_id} exceeded the 6 GiB RSS ceiling: {monitor.peak}")
        if interrupt_after_receipt:
            raise RuntimeError("synthetic interruption after durable fitted output and receipt")
        receipt_sha = sha256_file(receipt_path)
        _append_event({"event": "FIT_SAVED", "batch_id": BATCH_ID, "reservation_key": RESERVATION_KEY,
                       "card_id": CARD_ID, "stage": STAGE, "fit_id": fit_id, "cutoff": 79,
                       "component": component, "receipt_sha256": receipt_sha,
                       "amendment_id": None if synthetic else AMENDMENT_ID}, ledger_path=ledger_path, synthetic=synthetic)
        return {"status": "FIT_SAVED", "fit_id": fit_id, "receipt_sha256": receipt_sha,
                "artifact_sha256": sha256_file(artifact), "state_sha256": receipt["state_sha256"],
                "elapsed_seconds": receipt["elapsed_seconds"], "peak_rss_bytes": receipt["peak_rss_bytes"]}
    except Exception as exc:
        if interrupt_after_receipt and synthetic:
            raise
        existing = read_events(ledger_path)
        _starts, terminal = _fit_events(existing, fit_id)
        if not terminal:
            _append_event({"event": "FIT_FAILED", "batch_id": BATCH_ID, "reservation_key": RESERVATION_KEY,
                           "card_id": CARD_ID, "stage": STAGE, "fit_id": fit_id, "cutoff": 79,
                           "component": component, "error": f"{type(exc).__name__}: {exc}",
                           "amendment_id": None if synthetic else AMENDMENT_ID}, ledger_path=ledger_path, synthetic=synthetic)
        raise


def _state_summary(value: Any) -> Any:
    if hasattr(value, "audit_state") and callable(value.audit_state):
        return value.audit_state()
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in {"estimator", "preprocessor", "scaler", "encoder", "vectorizer", "model"}:
                if hasattr(item, "get_params"):
                    result[key] = {"class": type(item).__name__, "params": item.get_params(deep=False)}
                    if hasattr(item, "coef_"):
                        result[key]["coef_sha256"] = positional.sha256_array(item.coef_)
                    elif hasattr(item, "feature_importances_"):
                        result[key]["feature_importances_sha256"] = positional.sha256_array(item.feature_importances_)
                    if hasattr(item, "mean_"):
                        result[key]["mean_sha256"] = positional.sha256_array(item.mean_)
                else:
                    result[key] = {"class": type(item).__name__}
            elif isinstance(item, (np.ndarray, list, tuple)):
                try:
                    result[key] = {"shape": list(np.asarray(item).shape), "dtype": str(np.asarray(item).dtype),
                                   "sha256": positional.sha256_array(np.asarray(item))}
                except (TypeError, ValueError):
                    result[key] = str(item)
            elif isinstance(item, (str, int, float, bool)) or item is None:
                result[key] = item
            elif isinstance(item, Mapping):
                result[key] = _state_summary(item)
            else:
                result[key] = {"class": type(item).__name__}
        return result
    return {"class": type(value).__name__}


def load_train() -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["record_id", "artifact_signal", *CLASSIFICATION_FEATURES]
    frame = pd.read_csv(TRAIN, usecols=columns, dtype={"record_id": str})
    if len(frame) != 50_608 or frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise BatchError("Official Classification Train rows or IDs failed")
    target = pd.to_numeric(frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    if not np.array_equal(np.unique(target), np.asarray([0, 1], dtype=np.int8)):
        raise BatchError("Official Train artifact_signal is not binary 0/1")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if cycles.min() != 47 or cycles.max() != 79 or not np.array_equal(np.unique(cycles), np.arange(47, 80)):
        raise BatchError("Official Train cycle population differs from cycles 47-79")
    return frame, target


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, list(CLASSIFICATION_FEATURES)].copy()


def _c2b_vectorizer() -> tuple[Any, Any]:
    model = joblib.load(A02_MODEL)
    vectorizer = model.text_space.vectorizer
    params = positional._TFIDF_PARAMS
    fresh = __import__("sklearn.feature_extraction.text", fromlist=["TfidfVectorizer"]).TfidfVectorizer(**params)
    if (vectorizer.get_params(deep=False) != fresh.get_params(deep=False)
            or not vectorizer.vocabulary_ or len(vectorizer.vocabulary_) != 20_000
            or not np.isfinite(vectorizer.idf_).all()):
        raise BatchError("Reused A02 vectorizer parameters/vocabulary/IDF failed exact audit")
    train, _ = load_train()
    docs, missing = a02._documents(train.signal_stream.reset_index(drop=True))
    if (len(docs) != 50_608 or model.text_space.fit_rows != 50_608 or model.text_space.origin_cycle != 79
            or model.text_space.fit_documents_sha256 != a02._json_sha256(docs)
            or model.text_space.fit_row_positions_sha256 != positional.sha256_array(np.arange(50_608, dtype=np.int64))
            or not np.array_equal(missing, positional._text_values(train.signal_stream)[1])):
        raise BatchError("A02 E79 vectorizer fit population or documents do not match official Train")
    return vectorizer, model


def _c2b_matrices(frame: pd.DataFrame, *, vectorizer: Any | None = None,
                  train_side: bool) -> tuple[sparse.csr_matrix, sparse.csr_matrix | None, dict[str, Any]]:
    state = _load_unit(FIT_IDS[0])
    scaler = _load_unit(FIT_IDS[1])
    encoder = _load_unit(FIT_IDS[2])
    if vectorizer is None:
        vectorizer, _ = _c2b_vectorizer()
    selected = _features(frame)
    structured = positional.transform_c2b_structured(state, scaler, encoder, selected)
    text, missing = positional.transform_c2b_text(vectorizer, selected)
    base = sparse.hstack((structured, text, sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
                         format="csr", dtype=np.float64)
    if base.shape[1] != structured.shape[1] + len(vectorizer.vocabulary_) + 1:
        raise BatchError("C2_B feature block dimension/order mismatch")
    position = None
    if train_side:
        docs, _ = positional._text_values(selected.signal_stream)
        position, _audit = positional.build_position_block(docs, vectorizer, shuffled=False)
    return base, position, {"missing": missing, "structured_width": int(structured.shape[1]),
                            "text_width": int(text.shape[1]), "base_width": int(base.shape[1]),
                            "base_sha256": positional.sha256_csr(base),
                            "position_sha256": positional.sha256_csr(position) if position is not None else None}


def _fit_logistic(matrix: sparse.spmatrix, target: np.ndarray, *, label: str) -> tuple[LogisticRegression, dict[str, Any]]:
    model = LogisticRegression(**positional._LR_PARAMS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix, target)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence or not np.array_equal(model.classes_, np.asarray([0, 1])):
        raise BatchError(f"{label} failed convergence or binary class-order check: {convergence}")
    return model, {"estimator": "sklearn.linear_model.LogisticRegression",
                   "parameters": dict(positional._LR_PARAMS), "classes": model.classes_.tolist(),
                   "n_iter": model.n_iter_.tolist(), "convergence_warnings": convergence,
                   "coef_sha256": positional.sha256_array(model.coef_),
                   "intercept_sha256": positional.sha256_array(model.intercept_),
                   "fit_matrix_sha256": positional.sha256_csr(matrix)}


def _make_fit_payload(fit_id: str, frame: pd.DataFrame, target: np.ndarray) -> Any:
    features = _features(frame)
    if fit_id == FIT_IDS[0]:
        return positional.fit_c2b_structured_state(features)
    if fit_id == FIT_IDS[1]:
        return positional.fit_c2b_scaler(_load_unit(FIT_IDS[0]), features)
    if fit_id == FIT_IDS[2]:
        return positional.fit_c2b_ohe(_load_unit(FIT_IDS[0]), features)
    if fit_id == FIT_IDS[3]:
        vectorizer, _ = _c2b_vectorizer()
        matrix, _position, audit = _c2b_matrices(frame, vectorizer=vectorizer, train_side=False)
        model, meta = _fit_logistic(matrix, target, label="C2_B LogisticRegression")
        return {"estimator": model, "fit_metadata": meta,
                "structured_width": audit["structured_width"], "text_width": audit["text_width"],
                "base_width": audit["base_width"], "fit_rows": len(frame),
                "fit_matrix_sha256": audit["base_sha256"],
                "fit_ids_sha256": positional.sha256_array(frame.record_id.astype(str).to_numpy())}
    if fit_id == FIT_IDS[4]:
        return positional.fit_c1b_structured_state(features)
    if fit_id == FIT_IDS[5]:
        state = _load_unit(FIT_IDS[4])
        matrix = positional.transform_c1b_structured(state, features)
        model, meta = positional.fit_missing_hgb(state, matrix, target)
        return {"estimator": model, "fit_metadata": meta,
                "fit_matrix_sha256": positional.sha256_array(matrix),
                "fit_rows": len(frame),
                "fit_ids_sha256": positional.sha256_array(frame.record_id.astype(str).to_numpy())}
    if fit_id == FIT_IDS[6]:
        vectorizer, _ = _c2b_vectorizer()
        base, position, audit = _c2b_matrices(frame, vectorizer=vectorizer, train_side=True)
        assert position is not None
        bag = _load_unit(FIT_IDS[3])["estimator"]
        z0 = np.asarray(bag.decision_function(base), dtype=np.float64).reshape(-1)
        gamma, meta = __import__("src.models.positional_bag_offset_v1", fromlist=["fit_offset_coefficients"]).fit_offset_coefficients(
            position, z0, target, c=1.0)
        return {"gamma": gamma, "fit_metadata": meta,
                "fit_rows": len(frame), "fit_matrix_sha256": positional.sha256_csr(position),
                "base_matrix_sha256": audit["base_sha256"], "z0_sha256": positional.sha256_array(z0),
                "bag_lr_coefficient_sha256": positional.sha256_array(bag.coef_),
                "training_rows_include_missing_streams": True}
    if fit_id == FIT_IDS[7]:
        vectorizer, _ = _c2b_vectorizer()
        base, position, audit = _c2b_matrices(frame, vectorizer=vectorizer, train_side=True)
        assert position is not None
        joint = sparse.hstack((base, position), format="csr", dtype=np.float64)
        model, meta = _fit_logistic(joint, target, label="JOINTPOS LogisticRegression")
        return {"estimator": model, "fit_metadata": meta,
                "fit_rows": len(frame), "base_width": audit["base_width"],
                "position_width": int(position.shape[1]), "joint_width": int(joint.shape[1]),
                "fit_matrix_sha256": positional.sha256_csr(joint),
                "position_block_sha256": audit["position_sha256"]}
    if fit_id == FIT_IDS[8]:
        vectorizer, a02_model = _c2b_vectorizer()
        c2b = _load_unit(FIT_IDS[3])
        estimator = c2b["estimator"]
        start = int(c2b["structured_width"])
        width = len(vectorizer.vocabulary_)
        if estimator.coef_.shape[1] != start + width + 1:
            raise BatchError("C2_B coefficient width does not map structured/text/missing feature blocks")
        beta_a02 = np.empty(width, dtype=np.float64)
        for token, index in vectorizer.vocabulary_.items():
            beta_a02[int(index)] = float(estimator.coef_[0, start + int(index)])
        if not np.isfinite(beta_a02).all():
            raise BatchError("C2_B coefficient token map produced non-finite values")
        predictor = CoefficientGeometryPredictor(origin_cycle=79, arm="aligned").fit(
            text_space=a02_model.text_space,
            bank_matrix=a02_model.bank_matrix,
            bank_ids=a02_model.bank_ids,
            bank_row_positions=a02_model.bank_row_positions,
            bank_cycles=a02_model.bank_cycles,
            bank_oof_probabilities=a02_model.bank_oof_probabilities,
            bank_residuals=a02_model.bank_residuals,
            beta_in_a02_column_order=beta_a02,
        )
        return {"predictor": predictor, "fit_metadata": predictor.audit_state(),
                "fit_rows": 50_608, "beta_sha256": positional.sha256_array(beta_a02),
                "beta_token_count": int(width),
                "c2b_coefficient_sha256": positional.sha256_array(estimator.coef_),
                "a02_model_sha256": sha256_file(A02_MODEL),
                "token_vocabulary_sha256": positional.canonical_sha256({
                    "vocabulary": sorted((str(k), int(v)) for k, v in vectorizer.vocabulary_.items()),
                    "idf": positional.sha256_array(vectorizer.idf_),
                })}
    raise BatchError(f"No fit payload builder registered for {fit_id}")


def fit_next() -> dict[str, Any]:
    require_clean_tree()
    current_head = verify_preparation_head()
    input_hashes()
    if TRAINED_FREEZE.exists():
        raise BatchError("Unexpected trained freeze exists before fit completion")
    preflight_receipt = _read_json(PREFLIGHT_PATH)
    if (preflight_receipt.get("status") != "PASS_PREFIT_AND_REUSE_AUDIT"
            or preflight_receipt.get("preparation_head") != current_head
            or preflight_receipt.get("ledger", {}).get("sha256") != INITIAL_LEDGER_SHA
            or preflight_receipt.get("official_input_sha256") != input_hashes()
            or preflight_receipt.get("existing_model_reuse_audit") != _verify_reuse_artifacts()):
        raise BatchError("Durable preflight receipt no longer matches the exact source, inputs, or reuse audit")
    qa = _read_json(SYNTHETIC_QA)
    if (qa.get("status") != "PASS_SYNTHETIC_END_TO_END_AND_PERSISTENCE_RESUME"
            or qa.get("shared_ledger_sha256_after") != INITIAL_LEDGER_SHA
            or qa.get("source_sha256") != source_hashes()):
        raise BatchError("Required synthetic lifecycle/resume QA did not pass against the unchanged pre-fit ledger")
    budget_result = amend_and_reserve()
    events = read_events(LEDGER)
    current = _budget_snapshot(events)
    if current["fit_starts"] > 127:
        raise BatchError(f"Shared fit-start cap exceeded: {current}")
    starts_all = {e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"}
    failed = {e.get("fit_id") for e in events if e.get("event") == "FIT_FAILED"}
    if failed.intersection(FIT_IDS):
        raise BatchError(f"This batch already has a failed unit; no retries: {sorted(failed.intersection(FIT_IDS))}")
    pending_ids = []
    for fit_id in FIT_IDS:
        starts, terminals = _fit_events(events, fit_id)
        if starts and not terminals:
            pending_ids.append(fit_id)
    if pending_ids:
        fit_id = pending_ids[0]
        component = UNIT_COMPONENTS[FIT_IDS.index(fit_id)]
        with training_lock():
            recovery = execute_fit_unit(fit_id, component, lambda: None, synthetic=False, ledger_path=LEDGER)
        if recovery["status"] != "RECOVERED_DURABLE_RECEIPT":
            raise BatchError(f"Pending fit recovery did not produce a terminal receipt: {recovery}")
        return {"status": "PENDING_FIT_RECEIPT_RECOVERED_NO_REFIT", "fit": recovery,
                "budget": _batch_budget_status(read_events(LEDGER))}
    remaining = [fit_id for fit_id in FIT_IDS if fit_id not in starts_all]
    if not remaining:
        return {"status": "ALL_NINE_UNITS_ALREADY_STARTED", "budget": budget_result}
    fit_id = remaining[0]
    prep_status = _batch_budget_status(events)
    if prep_status["fit_pending"] != 0:
        raise BatchError(f"Another fit is pending; only one fit may run at a time: {prep_status}")
    index = FIT_IDS.index(fit_id)
    train, target = load_train()
    component = UNIT_COMPONENTS[index]
    with training_lock():
        result = execute_fit_unit(fit_id, component,
                                  lambda: _make_fit_payload(fit_id, train, target),
                                  synthetic=False, ledger_path=LEDGER)
    del train, target
    gc.collect()
    return {"status": "ONE_OFFICIAL_UNIT_COMPLETE", "fit": result,
            "remaining_fit_ids": [value for value in FIT_IDS if value != fit_id and value not in starts_all],
            "budget": _batch_budget_status(read_events(LEDGER))}


def _batch_budget_status(events: list[dict[str, Any]]) -> dict[str, Any]:
    amendment, reservation = _verify_batch_amendment(events)
    if amendment is None or reservation is None:
        raise BatchError("Batch cap amendment and nine-unit reservation must both exist")
    snapshot = _budget_snapshot(events)
    own = [event for event in events if event.get("reservation_key") == RESERVATION_KEY]
    starts = {event.get("fit_id") for event in own if event.get("event") == "FIT_STARTED"}
    saved = {event.get("fit_id") for event in own if event.get("event") == "FIT_SAVED"}
    failed = {event.get("fit_id") for event in own if event.get("event") == "FIT_FAILED"}
    if (reservation.get("fit_ids") != list(FIT_IDS) or reservation.get("fits") != 9
            or snapshot["global_cap"] != 130 or snapshot["category_caps"] != AMENDED_CATEGORY_CAPS
            or snapshot["active_reservations"] != 127
            or snapshot["active_reservations_by_category"].get("continuation") != 47
            or starts - set(FIT_IDS) or saved - starts or failed - starts
            or saved & failed or len(starts) > 9 or snapshot["fit_starts"] > 127):
        raise BatchError(f"Batch budget state differs from the nine-unit authorization: {snapshot}")
    expected_counts = {"fit_starts": 118 + len(starts), "fit_saved": 114 + len(saved),
                       "fit_failed": 4 + len(failed), "fit_pending": len(starts - saved - failed)}
    if any(snapshot[key] != value for key, value in expected_counts.items()):
        raise BatchError(f"Global fit event counts do not reconcile to this batch: {snapshot}")
    return snapshot


def fit_all() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    while True:
        events = read_events(LEDGER)
        started = {e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"}
        next_ids = [fit_id for fit_id in FIT_IDS if fit_id not in started]
        if not next_ids:
            break
        results.append(fit_next())
    return results


def freeze_trained() -> dict[str, Any]:
    require_clean_tree()
    events = read_events(LEDGER)
    own = [e for e in events if e.get("reservation_key") == RESERVATION_KEY]
    starts = {e.get("fit_id") for e in own if e.get("event") == "FIT_STARTED"}
    saved = {e.get("fit_id") for e in own if e.get("event") == "FIT_SAVED"}
    failed = {e.get("fit_id") for e in own if e.get("event") == "FIT_FAILED"}
    if starts != set(FIT_IDS) or saved != set(FIT_IDS) or failed:
        raise BatchError(f"Trained-state freeze requires exactly nine saved units: starts={len(starts)}, saved={len(saved)}, failed={len(failed)}")
    receipts = {fit_id: _verify_receipt(fit_id) for fit_id in FIT_IDS}
    source = source_hashes()
    inputs = input_hashes()
    result = {
        "schema_version": 1, "status": "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE",
        "batch_id": BATCH_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"), "source_sha256": source,
        "recipe_sha256": sha256_file(RECIPE_PATH), "amendment_sha256": sha256_file(BATCH_AMENDMENT_PATH),
        "official_inputs_sha256": inputs, "train_rows": 50_608, "train_cycles": [47, 79],
        "train_target_sha256": positional.sha256_array(load_train()[1]),
        "test_or_sample_rows_loaded": False, "test_labels_loaded": False,
        "fit_ids": list(FIT_IDS), "fit_receipts": {fit_id: sha256_file(_unit_receipt_path(fit_id)) for fit_id in FIT_IDS},
        "fitted_state_sha256": {fit_id: receipts[fit_id]["artifact_sha256"] for fit_id in FIT_IDS},
        "c2b_shared_components": list(FIT_IDS[:4]), "c1b_missing_fallback_components": list(FIT_IDS[4:6]),
        "pos_bag_offset_gamma": FIT_IDS[6], "jointpos_lr": FIT_IDS[7], "cgeom_predictor": FIT_IDS[8],
        "test_inference_started": False, "kaggle_contacted": False,
        "budget": _budget_snapshot(events),
    }
    write_json_once(TRAINED_FREEZE, result)
    return result


def _verify_trained_freeze() -> dict[str, Any]:
    freeze = _read_json(TRAINED_FREEZE)
    if (freeze.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or freeze.get("test_inference_started") is not False
            or freeze.get("test_labels_loaded") is not False
            or freeze.get("source_sha256") != source_hashes()
            or freeze.get("recipe_sha256") != sha256_file(RECIPE_PATH)
            or freeze.get("amendment_sha256") != sha256_file(BATCH_AMENDMENT_PATH)
            or freeze.get("fit_ids") != list(FIT_IDS)
            or freeze.get("fitted_state_sha256") != {fit_id: _verify_receipt(fit_id)["artifact_sha256"] for fit_id in FIT_IDS}
            or freeze.get("official_inputs_sha256") != input_hashes()):
        raise BatchError("Trained-state freeze does not match persisted full-Train components")
    return freeze


def _load_test_after_freeze() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    _verify_trained_freeze()
    train_rows = 50_608
    test = pd.read_csv(TEST, usecols=["record_id", *CLASSIFICATION_FEATURES], dtype={"record_id": str})
    sample = pd.read_csv(SAMPLE, dtype={"record_id": str})
    if len(test) != 29_421 or len(sample) != 29_421:
        raise BatchError("Official Test/SampleSubmission row counts must both be 29,421")
    test_ids = test.record_id.astype(str).to_numpy(dtype=str)
    sample_ids = sample.record_id.astype(str).to_numpy(dtype=str)
    if (test.record_id.isna().any() or test.record_id.duplicated().any()
            or not np.array_equal(test_ids, sample_ids)
            or sample.columns.tolist() != ["record_id", "artifact_signal"]):
        raise BatchError("Test IDs/order or SampleSubmission schema failed")
    if "artifact_signal" not in sample.columns:
        raise BatchError("SampleSubmission target column missing")
    baseline = c19._load_baseline_cache(test, sample)
    return test, sample, baseline


def _load_and_replay_a02(test: pd.DataFrame, baseline: Mapping[str, np.ndarray]) -> tuple[Any, dict[str, np.ndarray]]:
    model = joblib.load(A02_MODEL)
    with np.load(A02_TEST_NPZ, allow_pickle=False) as archive:
        saved = {key: np.asarray(archive[key]).copy() for key in archive.files}
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    rows = np.arange(len(test), dtype=np.int64) + 50_608
    result = model.predict(query_texts=test.signal_stream.reset_index(drop=True), query_record_ids=ids,
                           query_row_positions=rows, query_cycles=test.cycle_code.to_numpy(dtype=np.int64),
                           baseline_probabilities=baseline["probability"], missing_mask=baseline["missing_mask"])
    mapping = {
        "record_ids": ids,
        "baseline_probability": baseline["probability"],
        "missing_mask": baseline["missing_mask"],
        "candidate_probability": result["candidate_probability"],
        "labels": (result["candidate_probability"] >= THRESHOLD).astype(np.int8),
        "neighbor_count": result["neighbor_count"],
        "neighbor_source_rows": result["neighbor_source_rows"],
        "neighbor_ids": result["neighbor_ids"],
        "neighbor_similarities": result["neighbor_similarities"],
        "correction": result["correction"],
        "correction_unclipped": result["correction_raw"],
    }
    if set(saved) != set(mapping):
        raise BatchError(f"Saved A02 E79 Test prediction schema mismatch: {sorted(saved)}")
    mismatches = [key for key in mapping if not np.array_equal(saved[key], mapping[key])]
    if mismatches:
        raise BatchError(f"A02 E79 predictions failed exact feature-only replay: {mismatches}")
    return model, saved


def _positive_probability(estimator: Any, matrix: Any) -> np.ndarray:
    if not np.array_equal(np.asarray(estimator.classes_), np.asarray([0, 1])):
        raise BatchError("Classifier class order is not [0,1]")
    result = np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise BatchError("Classifier returned invalid probabilities")
    return result


def infer_three() -> dict[str, Any]:
    freeze = _verify_trained_freeze()
    require_clean_tree()
    with training_lock():
        test, sample, baseline = _load_test_after_freeze()
        a02_model, a02_saved = _load_and_replay_a02(test, baseline)
        vectorizer = a02_model.text_space.vectorizer
        test_base, test_position, c2b_audit = _c2b_matrices(test, vectorizer=vectorizer, train_side=True)
        if test_position is None:
            raise BatchError("Test positional transform was not produced")
        c2b_estimator = _load_unit(FIT_IDS[3])["estimator"]
        base_probability = _positive_probability(c2b_estimator, test_base)
        missing = np.asarray(c2b_audit["missing"], dtype=bool)
        if not np.array_equal(missing, baseline["missing_mask"]):
            raise BatchError("Reused A02 and C2_B test missing-stream mask differs")
        offset_artifact = _load_unit(FIT_IDS[6])
        offset_mod = __import__("src.models.positional_bag_offset_v1", fromlist=["offset_probability"])
        offset_raw = offset_mod.offset_probability(test_position,
                                                   np.asarray(c2b_estimator.decision_function(test_base)).reshape(-1),
                                                   offset_artifact["gamma"])
        c1_state = _load_unit(FIT_IDS[4])
        c1_estimator = _load_unit(FIT_IDS[5])["estimator"]
        c1_test = positional.transform_c1b_structured(c1_state, _features(test))
        missing_probability = positional.positive_probability(c1_estimator, c1_test)
        offset_probability = np.where(missing, missing_probability, offset_raw)

        joint_estimator = _load_unit(FIT_IDS[7])["estimator"]
        joint_matrix = sparse.hstack((test_base, test_position), format="csr", dtype=np.float64)
        joint_raw = _positive_probability(joint_estimator, joint_matrix)
        joint_probability = np.where(missing, missing_probability, joint_raw)

        cgeom = _load_unit(FIT_IDS[8])["predictor"]
        cgeom_output = cgeom.predict(
            query_texts=test.signal_stream.reset_index(drop=True),
            query_record_ids=test.record_id.astype(str).to_numpy(dtype=str),
            query_row_positions=np.arange(len(test), dtype=np.int64) + 50_608,
            query_cycles=test.cycle_code.to_numpy(dtype=np.int64),
            baseline_probabilities=baseline["probability"],
            original_a02_probabilities=a02_saved["candidate_probability"],
            original_a02_neighbor_source_rows=a02_saved["neighbor_source_rows"],
            original_a02_neighbor_similarities=a02_saved["neighbor_similarities"],
            missing_mask=baseline["missing_mask"],
        )
        cgeom_probability = cgeom_output["candidate_probability"]
        probability_map = {
            "CGEOM_ALIGNED": cgeom_probability,
            "POS_BAG_OFFSET": offset_probability,
            "JOINTPOS": joint_probability,
        }
        if np.any(test.cycle_code.to_numpy(dtype=np.int64) < 80) or np.any(test.cycle_code.to_numpy(dtype=np.int64) > 91):
            raise BatchError("Official Test cycle support must be exactly within 80-91")
        if not np.array_equal(np.unique(test.cycle_code.to_numpy(dtype=np.int64)), np.arange(80, 92, dtype=np.int64)):
            raise BatchError("Official Test must include all complete cycles 80-91")
        for name, probability in probability_map.items():
            if (probability.shape != (29_421,) or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise BatchError(f"Invalid output probabilities for {name}")
        if not np.array_equal(offset_probability[missing], missing_probability[missing]) or not np.array_equal(joint_probability[missing], missing_probability[missing]):
            raise BatchError("POS_BAG_OFFSET or JOINTPOS missing-stream route differs from C1_B HGB")
        if not np.array_equal(cgeom_probability[baseline["missing_mask"]], baseline["probability"][baseline["missing_mask"]]):
            raise BatchError("CGEOM missing-stream fallback differs from original A02 baseline")

        output_arrays: dict[str, np.ndarray] = {
            "record_ids": test.record_id.astype(str).to_numpy(dtype=str),
            "missing_mask": missing,
            "a02_baseline_probability": baseline["probability"].astype(np.float64),
            "a02_probability": a02_saved["candidate_probability"].astype(np.float64),
            "c2b_probability": base_probability.astype(np.float64),
            "c1b_missing_probability": missing_probability.astype(np.float64),
            "cgeom_aligned_probability": cgeom_probability.astype(np.float64),
            "posoffset_probability": offset_probability.astype(np.float64),
            "jointpos_probability": joint_probability.astype(np.float64),
            "cgeom_aligned_labels": (cgeom_probability >= THRESHOLD).astype(np.int8),
            "posoffset_labels": (offset_probability >= THRESHOLD).astype(np.int8),
            "jointpos_labels": (joint_probability >= THRESHOLD).astype(np.int8),
            "cgeom_neighbor_count": cgeom_output["neighbor_count"].astype(np.int16),
            "cgeom_neighbor_rows": cgeom_output["neighbor_source_rows"].astype(np.int64),
            "cgeom_neighbor_ids": cgeom_output["neighbor_ids"].astype(str),
            "cgeom_neighbor_similarities": cgeom_output["neighbor_similarities"].astype(np.float64),
            "cgeom_correction": cgeom_output["correction"].astype(np.float64),
        }
        if OUTPUT_NPZ.exists():
            raise BatchError("Refusing to overwrite the immutable Test prediction bundle")
        tmp_npz = OUTPUT_NPZ.with_name(OUTPUT_NPZ.name + f".{os.getpid()}.partial.npz")
        try:
            np.savez_compressed(tmp_npz, **output_arrays)
            os.link(tmp_npz, OUTPUT_NPZ)
        finally:
            tmp_npz.unlink(missing_ok=True)
        arrays_hash = sha256_file(OUTPUT_NPZ)
        ids = test.record_id.astype(str).to_numpy(dtype=str)
        csv_paths: dict[str, Path] = {
            "CGEOM_ALIGNED": OUTPUT_ROOT / "classification_cgeom_aligned_e79.csv",
            "POS_BAG_OFFSET": OUTPUT_ROOT / "classification_posoffset_e79.csv",
            "JOINTPOS": OUTPUT_ROOT / "classification_jointpos_e79.csv",
        }
        csv_details = {}
        labels_map = {
            "CGEOM_ALIGNED": output_arrays["cgeom_aligned_labels"],
            "POS_BAG_OFFSET": output_arrays["posoffset_labels"],
            "JOINTPOS": output_arrays["jointpos_labels"],
        }
        for candidate, path in csv_paths.items():
            if path.exists():
                raise BatchError(f"Refusing to overwrite existing CSV: {path}")
            frame = pd.DataFrame({"record_id": ids, "artifact_signal": labels_map[candidate].astype(np.int8)})
            if frame.columns.tolist() != sample.columns.tolist() or not np.isin(frame.artifact_signal, [0, 1]).all():
                raise BatchError(f"CSV schema/labels failed for {candidate}")
            payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            write_once(path, payload)
            reread = pd.read_csv(path, dtype={"record_id": str})
            if (len(reread) != 29_421 or reread.columns.tolist() != ["record_id", "artifact_signal"]
                    or reread.record_id.isna().any() or reread.record_id.duplicated().any()
                    or not np.array_equal(reread.record_id.to_numpy(dtype=str), sample.record_id.to_numpy(dtype=str))
                    or reread.artifact_signal.dtype.kind not in "iu"
                    or not np.isin(reread.artifact_signal.to_numpy(), [0, 1]).all()
                    or not np.array_equal(reread.artifact_signal.to_numpy(dtype=np.int8), labels_map[candidate])):
                raise BatchError(f"CSV read-back verification failed for {candidate}")
            csv_details[candidate] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path), "rows": len(reread), "columns": reread.columns.tolist(),
                "ordered_ids_sha256": positional.sha256_array(reread.record_id.to_numpy(dtype=str)),
                "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels_map[candidate], return_counts=True))},
            }
        sparse25_labels = (baseline["probability"] >= THRESHOLD).astype(np.int8)
        a02_labels = a02_saved["labels"].astype(np.int8)
        changes_vs_a02: dict[str, Any] = {}
        changes_vs_sparse25: dict[str, Any] = {}
        for candidate, labels in labels_map.items():
            delta = labels != a02_labels
            changes_vs_a02[candidate] = {"changed_labels": int(delta.sum()), "positive_to_negative": int(np.sum((a02_labels == 1) & (labels == 0))),
                                         "negative_to_positive": int(np.sum((a02_labels == 0) & (labels == 1))),
                                         "descriptive_only": True}
            sparse_delta = labels != sparse25_labels
            changes_vs_sparse25[candidate] = {"changed_labels": int(sparse_delta.sum()),
                                              "positive_to_negative": int(np.sum((sparse25_labels == 1) & (labels == 0))),
                                              "negative_to_positive": int(np.sum((sparse25_labels == 0) & (labels == 1))),
                                              "descriptive_only": True}
        pairwise: dict[str, Any] = {}
        names = list(labels_map)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                pairwise[f"{left}_vs_{right}"] = int(np.sum(labels_map[left] != labels_map[right]))
        receipt = {
            "status": "PREDICTIONS_AND_CSVS_CREATED_NOT_UPLOADED",
            "trained_freeze_sha256": sha256_file(TRAINED_FREEZE), "prediction_bundle_sha256": arrays_hash,
            "official_inputs_sha256": input_hashes(), "test_rows": len(test),
            "test_targets_loaded": False, "sample_submission_used_for_schema_and_order_only": True,
            "a02_replay_verified_bit_exact": True, "a02_prediction_hashes": {
                "probability_sha256": positional.sha256_array(a02_saved["candidate_probability"]),
                "label_sha256": positional.sha256_array(a02_saved["labels"]),
                "baseline_probability_sha256": positional.sha256_array(baseline["probability"]),
                "missing_mask_sha256": positional.sha256_array(baseline["missing_mask"]),
            },
            "cgeom_diagnostics": cgeom_output["diagnostics"], "test_c2b_matrix": c2b_audit,
            "csvs": csv_details, "label_changes_vs_a02": changes_vs_a02,
            "label_changes_vs_sparse25": changes_vs_sparse25,
            "pairwise_candidate_label_disagreements": pairwise,
            "used_test_for_model_selection": False, "kaggle_contacted": False,
        }
        output_receipt = RUN_ROOT / "inference_receipt.json"
        write_json_once(output_receipt, receipt)
        return receipt


def _array_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BatchError(f"Refusing to overwrite NPZ output: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(temp, **arrays)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays) or any(archive[name].dtype.kind == "O" for name in archive.files):
            raise BatchError("NPZ output schema or dtype failed allow_pickle=False validation")


def verify_outputs() -> dict[str, Any]:
    freeze = _verify_trained_freeze()
    receipt = _read_json(RUN_ROOT / "inference_receipt.json")
    if (receipt.get("status") != "PREDICTIONS_AND_CSVS_CREATED_NOT_UPLOADED"
            or receipt.get("trained_freeze_sha256") != sha256_file(TRAINED_FREEZE)
            or receipt.get("test_targets_loaded") is not False or receipt.get("kaggle_contacted") is not False):
        raise BatchError("Inference receipt lacks expected non-submission/source-boundary status")
    with np.load(OUTPUT_NPZ, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"record_ids", "missing_mask", "a02_baseline_probability", "a02_probability", "c2b_probability",
                "c1b_missing_probability", "cgeom_aligned_probability", "posoffset_probability", "jointpos_probability",
                "cgeom_aligned_labels", "posoffset_labels", "jointpos_labels", "cgeom_neighbor_count",
                "cgeom_neighbor_rows", "cgeom_neighbor_ids", "cgeom_neighbor_similarities", "cgeom_correction"}
    if set(arrays) != required:
        raise BatchError("Final prediction NPZ schema differs")
    test_ids = pd.read_csv(TEST, usecols=["record_id"], dtype={"record_id": str}).record_id.to_numpy(dtype=str)
    sample = pd.read_csv(SAMPLE, dtype={"record_id": str})
    if (len(test_ids) != 29_421 or not np.array_equal(arrays["record_ids"].astype(str), test_ids)
            or not np.array_equal(sample.record_id.to_numpy(dtype=str), test_ids)):
        raise BatchError("Persisted Test IDs do not exactly match official Test/Sample order")
    csvs = receipt["csvs"]
    candidate_keys = {"CGEOM_ALIGNED": ("cgeom_aligned_labels", "classification_cgeom_aligned_e79.csv"),
                      "POS_BAG_OFFSET": ("posoffset_labels", "classification_posoffset_e79.csv"),
                      "JOINTPOS": ("jointpos_labels", "classification_jointpos_e79.csv")}
    for candidate, (array_key, filename) in candidate_keys.items():
        path = OUTPUT_ROOT / filename
        frame = pd.read_csv(path, dtype={"record_id": str})
        if (sha256_file(path) != csvs[candidate]["sha256"] or frame.columns.tolist() != sample.columns.tolist()
                or len(frame) != 29_421 or frame.record_id.isna().any() or frame.record_id.duplicated().any()
                or not np.array_equal(frame.record_id.to_numpy(dtype=str), test_ids)
                or frame.artifact_signal.dtype.kind not in "iu"
                or not np.array_equal(frame.artifact_signal.to_numpy(dtype=np.int8), arrays[array_key])):
            raise BatchError(f"Independent CSV verification failed: {filename}")
    budget = _budget_snapshot(read_events(LEDGER))
    own = [event for event in read_events(LEDGER) if event.get("reservation_key") == RESERVATION_KEY]
    starts = {event.get("fit_id") for event in own if event.get("event") == "FIT_STARTED"}
    saved = {event.get("fit_id") for event in own if event.get("event") == "FIT_SAVED"}
    if (starts != set(FIT_IDS) or saved != set(FIT_IDS) or budget["fit_pending"] != 0
            or budget["fit_starts"] != 127 or budget["fit_saved"] != 123 or budget["fit_failed"] != 4):
        raise BatchError(f"Final fit ledger does not reconcile exactly: {budget}")
    return {"status": "PASS_THREE_CSVS_AND_BUDGET_VERIFIED", "csvs": csvs,
            "prediction_bundle_sha256": sha256_file(OUTPUT_NPZ), "fit_ids": list(FIT_IDS),
            "budget": budget, "test_targets_loaded": False, "kaggle_contacted": False,
            "trained_freeze_sha256": sha256_file(TRAINED_FREEZE)}


def write_report() -> dict[str, Any]:
    verification = verify_outputs()
    inference = _read_json(RUN_ROOT / "inference_receipt.json")
    freeze = _read_json(TRAINED_FREEZE)
    manifest = {
        "schema_version": 1, "status": "PREPARED_FOR_BATCH_REVIEW_NOT_SUBMITTED",
        "batch_id": BATCH_ID, "source_commit": freeze["source_commit"],
        "latest_local_commit_at_report_time": _git("rev-parse", "HEAD"),
        "trained_freeze_sha256": sha256_file(TRAINED_FREEZE),
        "pre_fit_recipe_sha256": sha256_file(RECIPE_PATH),
        "pre_fit_amendment_sha256": sha256_file(BATCH_AMENDMENT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "synthetic_qa_sha256": sha256_file(SYNTHETIC_QA),
        "reused_artifact_hashes": _read_json(PREFLIGHT_PATH)["existing_model_reuse_audit"],
        "official_inputs_sha256": freeze["official_inputs_sha256"],
        "fit_ids": list(FIT_IDS), "fit_receipts": freeze["fit_receipts"],
        "fitted_state_sha256": freeze["fitted_state_sha256"],
        "prediction_bundle_sha256": inference["prediction_bundle_sha256"],
        "csvs": inference["csvs"],
        "label_changes_vs_a02": inference["label_changes_vs_a02"],
        "label_changes_vs_sparse25": inference["label_changes_vs_sparse25"],
        "a02_prediction_hashes": inference["a02_prediction_hashes"],
        "pairwise_candidate_label_disagreements": inference["pairwise_candidate_label_disagreements"],
        "a02_bit_exact_replay": inference["a02_replay_verified_bit_exact"],
        "historical_verdicts_unchanged": {
            "CGEOM_V1": "CLOSED_GATE_FAIL; no gate or historical verdict change",
            "POS_BAG_OFFSET_V1": "original diagnostic gates unchanged",
            "POSITIONAL_TEXT_PILOT_V1": "original pilot verdict unchanged",
        },
        "submission": {"attempted": False, "status": "NOT_SUBMITTED", "public_score": None},
        "test_target_loaded": False, "used_test_for_model_selection": False,
        "verification": verification,
        "budget_ledger_sha256": sha256_file(LEDGER),
    }
    if MANIFEST_PATH.exists():
        if _read_json(MANIFEST_PATH) != manifest:
            raise BatchError("Existing batch manifest differs; refusing overwrite")
    else:
        write_json_once(MANIFEST_PATH, manifest)
    report_text = _render_report(manifest)
    write_once(REPORT_PATH, report_text.encode("utf-8"))
    return manifest


def _render_report(manifest: Mapping[str, Any]) -> str:
    csvs = manifest["csvs"]
    changes = manifest["label_changes_vs_a02"]
    pairs = manifest["pairwise_candidate_label_disagreements"]
    rows = ["# A02/CGEOM three-candidate E79 preparation report", "",
            "**Status: `PREPARED_FOR_BATCH_REVIEW_NOT_SUBMITTED`. No Kaggle API, CLI, browser, or upload was used.**", "",
            "## Prepared CSVs", "", "| Candidate | CSV | SHA-256 | Label count (0 / 1) | Changed labels vs A02 |", "|---|---|---|---:|---:|"]
    for candidate, info in csvs.items():
        counts = info["label_counts"]
        rows.append(f"| {candidate} | `{info['path']}` | `{info['sha256']}` | {counts.get('0',0)} / {counts.get('1',0)} | {changes[candidate]['changed_labels']} |")
    rows.extend(["", "All three files have 29,421 rows, columns `record_id,artifact_signal`, unique ordered Test IDs identical to SampleSubmission, and integer labels 0/1. The Test target column was not read. Label changes and candidate disagreement counts are descriptive integrity summaries only; they were not used to select a recipe.", "",
                 "## Frozen recipes and historical evidence", "",
                 "- **CGEOM_ALIGNED** preserves the exact aligned CGEOM recipe and Sparse25/A02 fallback. Its historical verdict remains `CLOSED_GATE_FAIL`: E75 W6 was −0.000536338 Macro F1 versus A02 and −0.000245142 versus the permuted arm; all 52 subgroup checks passed. This preparation does not convert that result to a pass.",
                 "- **POS_BAG_OFFSET** preserves the real-position third-block recipe, C2_B fixed logit, original sum-loss gamma objective, C=1 and L-BFGS-B options, and C1_B missing-stream fallback. The historical E67 H12 improvement of +0.001151 remained below the +0.003 threshold; the original E75 W6 result and gate remain unchanged.",
                 "- **JOINTPOS** preserves the original real-position joint LR recipe and C1_B missing-stream fallback. Historical E67 H12 was +0.000889 versus bag and E75 W6 was −0.000246; its previous verdict is unchanged.",
                 "- No test labels or public scores were used. No candidate is declared champion or qualified based on the generated files.", "",
                 "## Fit and source accounting", "",
                 f"The exact nine reserved units all have one saved receipt; the shared ledger ends at {manifest['verification']['budget']['fit_starts']} starts / {manifest['verification']['budget']['fit_saved']} saved / {manifest['verification']['budget']['fit_failed']} failed / {manifest['verification']['budget']['fit_pending']} pending. The four failed counts are historical. Global cap is 130; continuation cap is 47; all other category caps remain fixed. No retry was used.",
                 "The A02 E79 vectorizer, A02 residual bank, original A02 predictions, and Sparse25 baseline were reused after hash/lineage checks. C16 full-Train cached predictions were not treated as fitted estimator states. No second C2_B or C1_B training job ran per candidate.", "",
                 "## Verification", "",
                 f"- Independent output and fit-accounting verification: `{manifest['verification']['status']}`.",
                 f"- A02 E79 Test prediction replay: bit-exact; probability SHA `{manifest['a02_prediction_hashes']['probability_sha256']}`; label SHA `{manifest['a02_prediction_hashes']['label_sha256']}`.",
                 f"- Source commit used for fitted state: `{manifest['source_commit']}`.",
                 f"- Trained-state freeze SHA-256: `{manifest['trained_freeze_sha256']}`.",
                 f"- Prediction NPZ SHA-256: `{manifest['prediction_bundle_sha256']}`.",
                 "- Test target loaded: false. Kaggle contacted: false. Upload: none.", "",
                 "## Pairwise label disagreement counts", ""])
    for pair, count in pairs.items():
        rows.append(f"- `{pair}`: {count} rows.")
    rows.extend(["", "These are prediction agreement diagnostics only, not model evaluation. Stop for review before any upload or further experiment.", ""])
    return "\n".join(rows)


def synthetic_lifecycle() -> dict[str, Any]:
    """Small end-to-end state/route/NPZ and interrupted-resume self-check."""
    import tempfile

    temp = Path(tempfile.mkdtemp(prefix="a02_cgeom_batch_synthetic_"))
    ledger = temp / "synthetic_ledger.jsonl"
    synthetic_root = temp / "units"
    artifact = synthetic_root / f"{FIT_IDS[0]}.joblib"
    receipt_path = synthetic_root / f"{FIT_IDS[0]}.json"
    for path in (artifact, receipt_path):
        if path.exists():
            raise BatchError("Synthetic QA refuses to use official fitted output paths")
    reservation = {"event": "CARD_FITS_RESERVED", "reservation_key": RESERVATION_KEY,
                   "card_id": CARD_ID, "stage": STAGE, "category": "continuation",
                   "fits": 1, "fit_ids": [FIT_IDS[0]], "synthetic": True}
    _append_event(reservation, ledger_path=ledger, synthetic=True)
    calls = {"count": 0}

    def callback() -> dict[str, Any]:
        calls["count"] += 1
        return {"synthetic_vector": np.asarray([0.25, 0.5, 0.75], dtype=np.float64),
                "audit": {"class": "SyntheticState", "size": 3}}

    # Exercise the shared transaction logic against a temporary ledger only.
    try:
        execute_fit_unit(FIT_IDS[0], "SYNTHETIC_ONLY", callback, synthetic=True,
                         ledger_path=ledger, interrupt_after_receipt=True,
                         artifact_root=synthetic_root)
    except RuntimeError as exc:
        if "synthetic interruption" not in str(exc):
            raise
    recovered = execute_fit_unit(FIT_IDS[0], "SYNTHETIC_ONLY", callback, synthetic=True,
                                 ledger_path=ledger, artifact_root=synthetic_root)
    if calls["count"] != 1 or recovered["status"] != "RECOVERED_DURABLE_RECEIPT":
        raise BatchError("Synthetic durable resume unexpectedly refit the callback")
    ids = np.asarray(["synthetic-λ-短い", "synthetic-🌱-" + "長" * 192], dtype=str)
    arrays = {"record_ids": ids, "probability": np.asarray([0.25, 0.75], dtype=np.float64),
              "labels": np.asarray([0, 1], dtype=np.int8)}
    npz_path = temp / "output.npz"
    _array_npz_once(npz_path, arrays)
    with np.load(npz_path, allow_pickle=False) as archive:
        restored_ids = np.asarray(archive["record_ids"]).astype(str)
        if not np.array_equal(restored_ids, ids) or restored_ids.dtype.kind != "U":
            raise BatchError("Synthetic NPZ Unicode ID roundtrip failed")
    return {"result": recovered, "calls": calls, "temporary_ledger": ledger,
            "unicode_ids_npz_sha256": sha256_file(npz_path)}


def run_synthetic_qa() -> dict[str, Any]:
    require_clean_tree()
    events, _snapshot = verify_initial_budget()
    before = sha256_file(LEDGER)
    if before != INITIAL_LEDGER_SHA:
        raise BatchError("Synthetic QA begins only from the immutable initial shared ledger")
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "tests/test_optimization_v3_a02_cgeom_benchmark_v1.py"]
    completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = (completed.stdout + "\n" + completed.stderr).strip()
    after = sha256_file(LEDGER)
    if completed.returncode != 0 or before != after:
        raise BatchError("Synthetic QA failed or touched the production ledger:\n" + output[-4000:])
    result = {"status": "PASS_SYNTHETIC_END_TO_END_AND_PERSISTENCE_RESUME",
              "source_commit": _git("rev-parse", "HEAD"), "source_sha256": source_hashes(),
              "official_train_sha256": EXPECTED_TRAIN_SHA,
              "shared_ledger_sha256_before": before, "shared_ledger_sha256_after": after,
              "temporary_ledger_only": True, "official_fit_units_started": 0,
              "test_features_loaded": False, "test_labels_loaded": False,
              "pytest_command": cmd, "pytest_output": output[-1500:],
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_once(SYNTHETIC_QA, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "synthetic-qa", "reserve", "fit-next", "fit-all",
                                             "freeze-trained", "infer", "verify", "report"])
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight()
    elif args.action == "synthetic-qa":
        result = run_synthetic_qa()
    elif args.action == "reserve":
        preflight()
        qa = _read_json(SYNTHETIC_QA)
        if qa.get("status") != "PASS_SYNTHETIC_END_TO_END_AND_PERSISTENCE_RESUME":
            raise BatchError("Synthetic QA must pass before the shared cap amendment and reservation")
        result = amend_and_reserve()
    elif args.action == "fit-next":
        result = fit_next()
    elif args.action == "fit-all":
        result = fit_all()
    elif args.action == "freeze-trained":
        result = freeze_trained()
    elif args.action == "infer":
        result = infer_three()
    elif args.action == "verify":
        result = verify_outputs()
    else:
        result = write_report()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
