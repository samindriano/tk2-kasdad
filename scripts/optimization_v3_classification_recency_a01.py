"""Frozen two-origin A01 recency-weighted sparse Classification pilot.

Only official Classification Train is read. The runner uses the frozen C16
cache as a row-matched control bridge and performs exactly four new estimator
fits: RECENCY and SHUFFLED_WITHIN_CLASS at cutoffs 55 and 67.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from scripts.research_campaign_classification import load_train
from src.data.audit import sha256 as file_sha256
from src.models.classification_models import _number_array
from src.models.classification_round2_models import stream_missing_mask
from src.models.classification_recency_a01 import (
    CARD_ID, EXPERT_WEIGHT, FEATURE_NAMES, ORIGINS, PRESENT_BASELINE_WEIGHT,
    PARAMS, RecencySparseClassifier, SHUFFLE_SEED, THRESHOLD, _TFIDF_PARAMS, recency_weight_plan,
    route_probabilities,
)
from src.models.optimization_v3_batch1 import json_sha256, read_bundle
from src.models.research_next_budget import finish_fit, ledger_summary, read_events, release_unused, start_fit
from src.models.research_next_contract import canonical_sha256, publish_bundle, read_bundle as read_research_bundle
from src.utils.paths import task_files


STAGE = "A01_WEIGHTED_CLASSIFICATION_PILOT"
ORIGINS = (55, 67)
ARMS = ("RECENCY", "SHUFFLED_WITHIN_CLASS")
FIT_ORDER = ((55, "RECENCY"), (55, "SHUFFLED_WITHIN_CLASS"),
             (67, "RECENCY"), (67, "SHUFFLED_WITHIN_CLASS"))
THREADS = 2
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
CONFIG_DIR = ROOT / "experiments/optimization_v3/classification_recency_a01"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_classification_recency_a01"
REPORT_ROOT = ROOT / "reports/optimization_v3/classification_recency_a01"
PROTOCOL_PATH = CONFIG_DIR / "protocol.json"
FREEZE_PATH = CONFIG_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_ROOT / "official_preflight.json"
RESULTS_PATH = REPORT_ROOT / "results.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
AMENDMENT_ID = "A01_PILOT_REALLOCATION_2026_11_02"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
SYNTHETIC_REPORT = REPORT_ROOT / "synthetic_preflight_v2.json"
SYNTHETIC_VERIFICATION = RUN_ROOT / "synthetic_preflight_v2/independent_verification.json"
C16_FREEZE = ROOT / "experiments/optimization_v3/batch1/C16/freeze.json"
C16_RECIPE = ROOT / "experiments/optimization_v3/batch1/C16/recipe.json"
C16_CACHE = ROOT / "experiments/runs/optimization_v3_batch1/C16"
C16_DRIFT_REPORT = ROOT / "reports/optimization_v3/batch1/RECONCILIATION_01.md"

GATES = {
    "primary_gain_vs_sparse25_min": 0.0,
    "primary_gain_vs_shuffled_min": 0.0,
    "H12_gain_vs_sparse25_min": 0.0005,
    "H12_gain_vs_shuffled_min": 0.0002,
    "H12_55_max_harm_vs_sparse25": 0.001,
    "H12_67_max_harm_vs_sparse25": 0.001,
    "W1_max_harm_vs_sparse25": 0.003,
    "W4_max_harm_vs_sparse25": 0.003,
}
SUBGROUP_SUPPORT = {"minimum_n": 200, "minimum_each_class_n": 30}
SUBGROUP_GUARDS = {"macro_f1_harm_max": 0.01, "positive_recall_harm_max": 0.03}
RISK_REVIEW = {"macro_f1_harm_gt": 0.03, "positive_recall_harm_gt": 0.05}
VIEWS: dict[str, list[tuple[int, int, int]]] = {
    "primary": [(55, 56, 59), (67, 68, 71)],
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "H12_55": [(55, 56, 67)],
    "H12_67": [(67, 68, 79)],
    "H12_pooled": [(55, 56, 67), (67, 68, 79)],
}

CODE_FILES = (
    "src/models/classification_recency_a01.py",
    "scripts/optimization_v3_classification_recency_a01.py",
    "scripts/optimization_v3_classification_recency_a01_verify.py",
    "scripts/optimization_v3_a01_preflight.py",
    "scripts/optimization_v3_a01_preflight_verify.py",
    "tests/test_optimization_v3_a01_preflight.py",
    "scripts/research_campaign_classification.py",
    "scripts/optimization_v3_batch1_c16.py",
    "scripts/optimization_v3_batch1_verify.py",
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/optimization_v3_sparse_classification.py",
    "src/models/optimization_v3_batch1_c16.py",
    "src/models/optimization_v3_batch1.py",
    "src/models/research_next_budget.py",
    "src/models/research_next_contract.py",
    "src/models/regression_research_text.py",
    "src/models/round1_models.py",
    "src/data/audit.py",
    "src/utils/paths.py",
)
STATIC_INPUTS = (
    "experiments/optimization_v3/batch1/C16/recipe.json",
    "experiments/optimization_v3/batch1/C16/freeze.json",
    "reports/optimization_v3/batch1/C16/results.json",
    "reports/optimization_v3/batch1/C16/verification.json",
    "reports/optimization_v3/batch1/C16/gate.json",
    "reports/optimization_v3/batch1/C16/summary.json",
    "reports/optimization_v3/batch1/RECONCILIATION_01.md",
    "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
    "experiments/optimization_v3/classification_sparse/freeze.json",
    "reports/classification_round1/fit_audits/fit55.json",
    "reports/classification_round1/fit_audits/fit67.json",
    "reports/optimization_v3/classification_recency_a01/synthetic_preflight_v2.json",
    "experiments/runs/optimization_v3_classification_recency_a01/synthetic_preflight_v2/independent_verification.json",
)


def _sha(path: Path) -> str:
    return file_sha256(Path(path))


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Immutable A01 output differs; refusing overwrite: {path}")
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
                          check=False, capture_output=True).returncode == 0


def _sha_int64(values: Any) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    frame, target, path = load_train(None)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (_sha(path) != TRAIN_SHA256 or len(frame) != 50_608 or frame["record_id"].isna().any()
            or not frame["record_id"].astype(str).is_unique
            or not np.array_equal(np.unique(cycles), np.arange(47, 80))
            or target.shape != (len(frame),) or not np.isin(target, (0, 1)).all()):
        raise RuntimeError("Official Classification Train failed A01 row/schema/hash checks")
    if set(FEATURE_NAMES) - set(frame.columns) or {"record_id", "artifact_signal", "threat_rank"} & set(FEATURE_NAMES):
        raise RuntimeError("A01 feature allowlist is missing or includes identity/target fields")
    return Path(path).resolve(), frame, np.asarray(target, dtype=np.int8), cycles


def _verify_c16_source_drift(frozen: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = frozen.get("identity", {}).get("source_sha256", {})
    relative = "src/models/optimization_v3_batch1.py"
    pinned_path = str((ROOT / relative).resolve())
    matches = [(Path(path), digest) for path, digest in source_hashes.items()
               if Path(path).resolve() == Path(pinned_path)]
    if len(matches) != 1:
        raise RuntimeError("C16 frozen source manifest is missing its shared helper identity")
    pinned_sha = str(matches[0][1])
    source_path = ROOT / relative
    current_sha = _sha(source_path)
    other_mismatches = []
    for path_text, expected in source_hashes.items():
        path = Path(path_text)
        if not path.is_file():
            other_mismatches.append(str(path))
        elif _sha(path) != expected and path.resolve() != source_path.resolve():
            other_mismatches.append(str(path))
    if other_mismatches or pinned_sha != "d21ac914d0d1f516a93e431116556dabdbe7313a21fd2b1b04a73371bd56f420" or current_sha != "746e12d408e74608e5b559bc7dcf4c10174df9f367859750987310adb6ae8ffd":
        raise RuntimeError(f"C16 source drift is outside the reconciled budget helper change: {other_mismatches}")
    old = subprocess.check_output(["git", "-C", str(ROOT), "show", f"9e9b7fb:{relative}"])
    if hashlib.sha256(old).hexdigest() != pinned_sha:
        raise RuntimeError("C16 historical helper blob does not match the source hash pinned in its freeze")
    def without_reservation_function(payload: bytes) -> str:
        module = ast.parse(payload.decode("utf-8"))
        module.body = [node for node in module.body
                       if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       or node.name != "reserve_card_budget"]
        return ast.dump(module, include_attributes=False)
    current = source_path.read_bytes()
    if without_reservation_function(old) != without_reservation_function(current):
        raise RuntimeError("C16 shared helper drift includes changes outside reserve_card_budget")
    report = C16_DRIFT_REPORT.read_text(encoding="utf-8")
    required = ("only later helper diff is budget-reservation bookkeeping in `reserve_card_budget`",
                pinned_sha, "current SHA `746e12d4")
    if not all(token in report for token in required):
        raise RuntimeError("C16 source-drift reconciliation report lacks its metadata-only conclusion")
    return {"status": "METADATA_ONLY_RESERVATION_BOOKKEEPING_ONLY",
            "path": relative, "pinned_sha256": pinned_sha, "current_sha256": current_sha,
            "historical_source_commit": "9e9b7fb",
            "changed_function": "reserve_card_budget",
            "ast_outside_changed_function_identical": True,
            "reconciliation_report": "reports/optimization_v3/batch1/RECONCILIATION_01.md",
            "reconciliation_report_sha256": _sha(C16_DRIFT_REPORT)}


def _read_c16_controls(frame: pd.DataFrame, cycles: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    from src.models.optimization_v3_batch1_c16 import PARAMS as C16_PARAMS

    frozen = _json(C16_FREEZE)
    recipe = _json(C16_RECIPE)
    identity = frozen.get("identity")
    if not isinstance(identity, dict) or frozen.get("signature") != json_sha256(identity):
        raise RuntimeError("C16 freeze signature is invalid")
    c16_source_drift = _verify_c16_source_drift(frozen)
    if identity.get("recipe_artifact_sha256") != json_sha256(recipe):
        raise RuntimeError("C16 frozen recipe does not match its recipe artifact hash")
    result: dict[int, dict[str, np.ndarray]] = {}
    fit_expected = {55: 12_020, 67: 29_705}
    for cutoff in ORIGINS:
        path = C16_CACHE / f"C16_fit{cutoff}.npz"
        score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        expected_identity = {"freeze_signature": frozen["signature"], "card_id": "C16",
                             "cutoff": cutoff, "score_rows": score_rows.tolist()}
        arrays, sidecar = read_bundle(path, expected_identity)
        metadata = sidecar.get("metadata", {})
        state = metadata.get("state", {})
        expected_fit = int(np.sum(cycles <= cutoff))
        expected_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy()
        required = {"score_rows", "score_record_ids", "prediction", "baseline_prediction",
                    "raw_candidate", "control__baseline_sparse_raw", "control__c2b_probability",
                    "control__cb_probability", "control__stream_missing", "control__control_rows"}
        if not required.issubset(arrays):
            raise RuntimeError(f"C16 cache {cutoff} is missing arrays: {sorted(required - set(arrays))}")
        checks = (
            len(score_rows) == (17_685 if cutoff == 55 else 20_903),
            np.array_equal(arrays["score_rows"], score_rows),
            np.array_equal(arrays["control__control_rows"], score_rows),
            np.array_equal(arrays["score_record_ids"].astype(str), expected_ids),
            int(metadata.get("fit_predictor_count", -1)) == 1,
            int(metadata.get("fit_rows", -1)) == fit_expected[cutoff] == expected_fit,
            int(metadata.get("score_rows", -1)) == len(score_rows),
            state.get("id") == "C16_MODERATE_SPARSE_CAPACITY_LGBM",
            int(state.get("cutoff", -1)) == cutoff,
            state.get("lightgbm_parameters") == dict(C16_PARAMS),
            state.get("sample_weight_used") is False,
            metadata.get("fit_metadata", {}).get("score_labels_passed_to_adapter") is False,
            set(arrays).isdisjoint({"artifact_signal", "target", "labels", "y"}),
        )
        if not all(checks):
            raise RuntimeError(f"C16 saved control identity/state checks failed at cutoff {cutoff}")
        missing = stream_missing_mask(frame.iloc[score_rows]).astype(bool)
        if not np.array_equal(arrays["control__stream_missing"].astype(bool), missing):
            raise RuntimeError(f"C16 availability mask differs from official Train at {cutoff}")
        raw_sparse = arrays["control__baseline_sparse_raw"].astype(np.float64)
        c2b = arrays["control__c2b_probability"].astype(np.float64)
        cb = arrays["control__cb_probability"].astype(np.float64)
        raw_c16 = arrays["raw_candidate"].astype(np.float64)
        baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * raw_sparse)
        candidate_c16 = np.where(missing, cb, 0.75 * c2b + 0.25 * raw_c16)
        np.testing.assert_allclose(baseline, arrays["baseline_prediction"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(candidate_c16, arrays["prediction"], rtol=0, atol=1e-12)
        for name, values in (("sparse25", baseline), ("C2_B", c2b), ("CB", cb)):
            if (values.shape != score_rows.shape or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))):
                raise RuntimeError(f"C16 control {name} is invalid at cutoff {cutoff}")
        result[cutoff] = {
            "rows": score_rows.astype(np.int64), "score_record_ids": expected_ids,
            "cycle_code": cycles[score_rows], "sparse25_raw": raw_sparse,
            "baseline": baseline.astype(np.float64), "c2b": c2b,
            "cb": cb, "missing": missing,
            "cache_npz_sha256": _sha(path), "cache_sidecar_sha256": _sha(path.with_suffix(".json")),
            "cache_receipt_integrity_sha256": sidecar["integrity_sha256"],
        }
    return result


def _budget_baseline_snapshot(*, require_original_state: bool) -> dict[str, Any]:
    raw = LEDGER.read_bytes()
    events = read_events(LEDGER)
    authorizations = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(authorizations) != 1:
        raise RuntimeError("Research Next budget ledger must have one original authorization")
    auth = authorizations[0]
    if (auth.get("batch_id") != "optimization_v3_research_next_v1" or int(auth.get("cap", -1)) != 60
            or auth.get("category_caps") != {"pilot": 26, "continuation": 24,
                                               "control_replay": 6, "technical_retry": 4}):
        raise RuntimeError("Research Next original allowance differs from the user-authorized 60-fit cap")
    releases: dict[str, int] = {}
    for event in events:
        if event.get("event") == "RESERVATION_UNUSED_RELEASED":
            key = str(event.get("reservation_key"))
            if key in releases:
                raise RuntimeError(f"Duplicate release event for {key}")
            releases[key] = int(event.get("unused_fits", 0))
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    active_by_category: dict[str, int] = {}
    active_by_key: dict[str, int] = {}
    for event in reservations:
        key = str(event["reservation_key"])
        fits = int(event["fits"])
        unused = releases.get(key, 0)
        if fits < 0 or unused < 0 or unused > fits:
            raise RuntimeError(f"Invalid active reservation/release arithmetic for {key}")
        active_by_key[key] = fits - unused
        category = str(event["category"])
        active_by_category[category] = active_by_category.get(category, 0) + fits - unused
    active_by_category = {str(category): active_by_category.get(str(category), 0)
                          for category in auth["category_caps"]}
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    snapshot = {
        "event_rows": len(events), "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "authorization_sha256": json_sha256(auth), "cap": 60,
        "category_caps": dict(auth["category_caps"]), "active_reserved": sum(active_by_key.values()),
        "active_by_category": active_by_category, "starts": len(starts),
        "saved": len(saved), "failed": len(failed),
        "active_by_key": active_by_key,
    }
    if require_original_state:
        expected = {
            "event_rows": 131,
            "ledger_sha256": "2b372c1a8aaf4adea51c9421800412279e8dced6180ac462a9c109d80fb9be1c",
            "active_reserved": 50,
            "active_by_category": {"continuation": 23, "pilot": 26,
                                   "control_replay": 0, "technical_retry": 1},
            "starts": 50, "saved": 48, "failed": 2,
        }
        if any(snapshot.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Live ledger differs from reviewed pre-amendment A01 checkpoint: {snapshot}")
    return snapshot


def _input_paths(train_path: Path) -> dict[str, Path]:
    paths = {"official_classification_train": train_path,
             **{name.replace("/", "__"): ROOT / name for name in STATIC_INPUTS}}
    for cutoff in ORIGINS:
        for suffix in ("npz", "json"):
            path = C16_CACHE / f"C16_fit{cutoff}.{suffix}"
            paths[f"C16_fit{cutoff}_{suffix}"] = path
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"A01 frozen input missing: {name}={path}")
    return paths


def _memory_estimate(fit_features: pd.DataFrame) -> dict[str, int]:
    from src.models.classification_models import _text_values
    docs, _missing = _text_values(fit_features["signal_stream"])
    token_counts = np.fromiter((len(re.findall(r"(?u)\b\w\w+\b", value.lower())) for value in docs),
                               dtype=np.int64, count=len(docs))
    nnz_upper = int(np.sum(token_counts + np.maximum(token_counts - 1, 0), dtype=np.int64))
    sparse_upper_bytes = int(nnz_upper * 16 + (len(docs) + 1) * 8)
    estimated_peak = int(1.5 * 1024**3 + 3 * sparse_upper_bytes)
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_MEMORY_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    return {"fit_rows": int(len(docs)), "estimated_nnz_upper_bound": nnz_upper,
            "estimated_sparse_bytes_upper_bound": sparse_upper_bytes,
            "estimated_peak_bytes": estimated_peak, "available_bytes": available,
            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
            "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
            "threads": THREADS, "concurrent_training_jobs": 1,
            "status": "PASS" if estimated_peak <= allowed else "BLOCKED_MEMORY"}


def _package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl", "psutil")}


def _protocol_identity(source_commit: str, *, budget_snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    train_path, frame, target, cycles = _load_train()
    controls = _read_c16_controls(frame, cycles)
    input_paths = _input_paths(train_path)
    input_hashes = {name: _sha(path) for name, path in sorted(input_paths.items())}
    code_hashes = {name: _sha(ROOT / name) for name in CODE_FILES}
    weight_diagnostics: dict[str, Any] = {}
    fit_rows_expected: dict[str, int] = {}
    score_rows_expected: dict[str, int] = {}
    memory: dict[str, Any] = {}
    for cutoff in ORIGINS:
        fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
        if len(fit_rows) != (12_020 if cutoff == 55 else 29_705):
            raise RuntimeError(f"A01 official fit population differs at cutoff {cutoff}")
        weights = recency_weight_plan(cycles[fit_rows], target[fit_rows], cutoff)
        weight_diagnostics[str(cutoff)] = weights["diagnostics"]
        fit_rows_expected[str(cutoff)] = int(len(fit_rows))
        score_rows_expected[str(cutoff)] = int(len(controls[cutoff]["rows"]))
        memory[str(cutoff)] = _memory_estimate(frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)])
    if not all(value["status"] == "PASS" for value in memory.values()):
        raise RuntimeError(f"A01 memory guard fails before freeze: {memory}")
    budget = dict(budget_snapshot) if budget_snapshot is not None else _budget_baseline_snapshot(require_original_state=True)
    c16 = _json(C16_FREEZE)
    return {
        "schema_version": 1, "campaign": CARD_ID, "stage": STAGE,
        "source_commit": source_commit, "official_train_rows": int(len(frame)),
        "official_train_sha256": input_hashes["official_classification_train"],
        "target": "artifact_signal", "score_data": "official Classification Train cycles after each frozen cutoff",
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
        "code_sha256": code_hashes, "input_sha256": input_hashes,
        "budget_pre_amendment": budget,
        "c16_source_drift_reconciliation": _verify_c16_source_drift(_json(C16_FREEZE)),
        "budget_amendment": {
            "amendment_id": AMENDMENT_ID, "source_category": "control_replay",
            "destination_category": "pilot", "fits": 4,
            "original_caps": {"continuation": 24, "control_replay": 6, "pilot": 26,
                              "technical_retry": 4},
            "effective_caps": {"continuation": 24, "control_replay": 2, "pilot": 30,
                               "technical_retry": 4}, "global_cap": 60,
        },
        "recipe": {
            "fit_origins": list(ORIGINS), "arms": list(ARMS),
            "fit_order": [list(x) for x in FIT_ORDER], "official_estimator_fits": 4,
            "fit_rows_expected": fit_rows_expected, "score_rows_expected": score_rows_expected,
            "feature_allowlist": list(FEATURE_NAMES),
            "model": {"class": "src.models.classification_recency_a01.RecencySparseClassifier",
                      "lightgbm_parameters": dict(PARAMS),
                      "tfidf_parameters": {k: list(v) if isinstance(v, tuple) else (v.__name__ if isinstance(v, type) else v)
                                           for k, v in _TFIDF_PARAMS.items()},
                      "structured_preprocessing": "RegressionTextModel structured transform, fitted only on current cutoff rows",
                      "early_stopping": False, "class_weight": None, "imbalance_adjustment": None,
                      "sample_weight_passed_only_to_estimator": True},
            "weights": {"formula": "a_i=max(0.25, 2^(-(cutoff-cycle_i)/8)); normalize to within-class mean 1",
                        "floor": 0.25, "half_life_cycles": 8.0,
                        "shuffled_control": "default_rng(42), reset per origin; class 0 then class 1; stable Train order",
                        "ess_ratio_each_class_min": 0.50,
                        "diagnostics_by_cutoff": weight_diagnostics},
            "availability_route": {
                "present_candidate": "0.75*frozen C2_B + 0.25*new RECENCY expert",
                "present_matched_control": "0.75*frozen C2_B + 0.25*new SHUFFLED_WITHIN_CLASS expert",
                "present_sparse25_baseline": "0.75*frozen C2_B + 0.25*frozen sparse25 expert",
                "missing_stream": "frozen CB probability for all arms",
                "threshold": THRESHOLD,
            },
            "memory_preflight": memory,
            "threads": THREADS, "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
            "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES, "concurrent_training_jobs": 1,
            "c16_freeze_signature": c16.get("signature"),
        },
        "evaluation": {
            "metric": "Macro F1, fixed threshold 0.50, labels [0,1], zero_division=0",
            "views": {name: [list(block) for block in blocks] for name, blocks in VIEWS.items()},
            "minimum_view_denominators": {"primary": 9_064, "W1": 4_114, "W4": 4_950,
                                          "H12_55": 17_685, "H12_67": 20_903,
                                          "H12_pooled": 38_588},
            "gates": GATES, "subgroup_support": SUBGROUP_SUPPORT,
            "subgroup_guards": SUBGROUP_GUARDS, "risk_review_hold": RISK_REVIEW,
            "subgroup_fields": ["fit_defined_text_load_quartile", "stream_availability", "entry_friction"],
        },
        "runtime": {"python": platform.python_version(), "packages": _package_versions()},
    }


def prepare() -> dict[str, Any]:
    if not SYNTHETIC_REPORT.is_file() or not SYNTHETIC_VERIFICATION.is_file():
        raise RuntimeError("A01 synthetic V2 report and independent verification must exist before official preparation")
    synthetic = _json(SYNTHETIC_REPORT)
    verification = _json(SYNTHETIC_VERIFICATION)
    if (synthetic.get("status") != "SYNTHETIC_PREFLIGHT_V2_PASS"
            or verification.get("status") != "PASS"
            or not all(verification.get("checks", {}).values())
            or verification.get("official_data_read") is not False
            or verification.get("official_fits_started") != 0
            or verification.get("test_read") is not False
            or verification.get("sample_submission_read") is not False
            or verification.get("kaggle_access") is not False):
        raise RuntimeError("A01 synthetic preflight/independent verification is not passing or violates source boundary")
    snapshot = _budget_baseline_snapshot(require_original_state=True)
    identity = _protocol_identity(_git_head(), budget_snapshot=snapshot)
    protocol_text = json.dumps(identity, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    signature = canonical_sha256(identity)
    freeze = {"campaign": CARD_ID, "identity": identity, "freeze_signature": signature,
              "protocol_sha256": hashlib.sha256(protocol_text.encode("utf-8")).hexdigest(),
              "freeze_status": "FROZEN_NO_OFFICIAL_FITS_STARTED"}
    _write_once(PROTOCOL_PATH, identity)
    _write_once(FREEZE_PATH, freeze)
    preflight = {"status": "OFFICIAL_PREFLIGHT_PASS", "campaign": CARD_ID,
                 "freeze_signature": signature, "protocol_sha256": freeze["protocol_sha256"],
                 "official_train_rows": identity["official_train_rows"],
                 "official_train_sha256": identity["official_train_sha256"],
                 "fit_rows_expected": identity["recipe"]["fit_rows_expected"],
                 "score_rows_expected": identity["recipe"]["score_rows_expected"],
                 "weight_ess": {key: value["by_class"] for key, value in
                                identity["recipe"]["weights"]["diagnostics_by_cutoff"].items()},
                 "c16_cache_sha256": {key: {"npz": controls["cache_npz_sha256"],
                                             "sidecar": controls["cache_sidecar_sha256"]}
                                      for key, controls in _read_c16_controls(*_load_train()[1::2]).items()},
                 "memory_preflight": identity["recipe"]["memory_preflight"],
                 "budget_pre_amendment": snapshot, "official_fits_started": 0,
                 "test_read": False, "sample_submission_read": False, "kaggle_access": False}
    _write_once(PREFLIGHT_PATH, preflight)
    return preflight


def _load_frozen() -> tuple[dict[str, Any], str]:
    if not FREEZE_PATH.is_file() or not PROTOCOL_PATH.is_file():
        raise RuntimeError("A01 protocol/freeze missing; run prepare after the source-code commit")
    frozen = _json(FREEZE_PATH)
    identity = frozen.get("identity")
    if not isinstance(identity, dict) or frozen.get("freeze_signature") != canonical_sha256(identity):
        raise RuntimeError("A01 freeze signature is invalid")
    if not _git_is_ancestor(str(identity.get("source_commit", ""))):
        raise RuntimeError("Current Git HEAD does not descend from the A01 code freeze commit")
    protocol = _json(PROTOCOL_PATH)
    protocol_bytes = (json.dumps(protocol, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if protocol != identity or hashlib.sha256(protocol_bytes).hexdigest() != frozen.get("protocol_sha256"):
        raise RuntimeError("A01 frozen protocol content/hash differs")
    current_code = {name: _sha(ROOT / name) for name in CODE_FILES}
    if current_code != identity.get("code_sha256"):
        raise RuntimeError("A01 code or frozen dependency hashes changed after freeze")
    train_path, frame, target, cycles = _load_train()
    controls = _read_c16_controls(frame, cycles)
    current_inputs = {name: _sha(path) for name, path in sorted(_input_paths(train_path).items())}
    if current_inputs != identity.get("input_sha256"):
        raise RuntimeError("A01 Train, C16, or preflight input hashes changed after freeze")
    if _package_versions() != identity.get("runtime", {}).get("packages"):
        raise RuntimeError("A01 runtime package versions differ from the frozen protocol")
    return identity, frozen["freeze_signature"]


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    encoded = (json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _effective_budget(events: list[dict[str, Any]], *, require_reservation: bool) -> dict[str, Any]:
    authorizations = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    amendments = [e for e in events if e.get("event") == "BUDGET_CATEGORY_AMENDMENT"
                  and e.get("amendment_id") == AMENDMENT_ID]
    if len(authorizations) != 1 or len(amendments) != 1:
        raise RuntimeError("A01 requires one original allowance and one exact category-amendment event")
    auth, amendment = authorizations[0], amendments[0]
    if amendment.get("original_authorization_sha256") != json_sha256(auth):
        raise RuntimeError("A01 amendment does not bind to the unchanged original authorization")
    if (amendment.get("source_category") != "control_replay" or amendment.get("destination_category") != "pilot"
            or int(amendment.get("fits", -1)) != 4 or amendment.get("effective_caps") != {
                "continuation": 24, "control_replay": 2, "pilot": 30, "technical_retry": 4}
            or amendment.get("original_caps") != {"continuation": 24, "control_replay": 6,
                                                   "pilot": 26, "technical_retry": 4}
            or int(amendment.get("global_cap", -1)) != 60):
        raise RuntimeError("A01 category amendment differs from the authorized four-slot transfer")
    reservation_events = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                          and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservation_events) > 1:
        raise RuntimeError("A01 reservation key is duplicated")
    if require_reservation and (len(reservation_events) != 1 or int(reservation_events[0].get("fits", -1)) != 4
                                or reservation_events[0].get("category") != "pilot"
                                or reservation_events[0].get("amendment_id") != AMENDMENT_ID):
        raise RuntimeError("A01 four-fit pilot reservation is absent or inconsistent")
    releases: dict[str, int] = {}
    for event in events:
        if event.get("event") == "RESERVATION_UNUSED_RELEASED":
            key = str(event.get("reservation_key"))
            if key in releases:
                raise RuntimeError(f"Duplicate unused-fit release for {key}")
            releases[key] = int(event.get("unused_fits", 0))
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    active_by_category: dict[str, int] = {}
    active_by_key: dict[str, int] = {}
    for event in reservations:
        key, gross = str(event["reservation_key"]), int(event["fits"])
        unused = releases.get(key, 0)
        if unused < 0 or unused > gross:
            raise RuntimeError(f"Invalid release quantity for {key}")
        active_by_key[key] = gross - unused
        cat = str(event["category"])
        active_by_category[cat] = active_by_category.get(cat, 0) + gross - unused
    caps = amendment["effective_caps"]
    active_by_category = {str(category): active_by_category.get(str(category), 0) for category in caps}
    total = sum(active_by_key.values())
    if total > 60 or any(active_by_category.get(cat, 0) > int(cap) for cat, cap in caps.items()):
        raise RuntimeError(f"A01 live effective budget cap exceeded: {total}/60 {active_by_category}/{caps}")
    if require_reservation and (total != 54 or active_by_category.get("pilot") != 30
                                or active_by_category.get("continuation") != 23
                                or active_by_category.get("control_replay") != 0
                                or active_by_category.get("technical_retry") != 1):
        raise RuntimeError(f"A01 post-reservation accounting differs from reviewed 54/60 checkpoint: {active_by_category}")
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    return {"active_reserved": total, "active_by_category": active_by_category,
            "started": len(starts), "saved": sum(e.get("event") == "FIT_SAVED" for e in terminals),
            "failed": sum(e.get("event") == "FIT_FAILED" for e in terminals),
            "category_caps": caps, "global_cap": 60,
            "amendment_id": AMENDMENT_ID, "reservation_key": RESERVATION_KEY}


def append_amendment() -> dict[str, Any]:
    identity, signature = _load_frozen()
    events = read_events(LEDGER)
    existing = [e for e in events if e.get("event") == "BUDGET_CATEGORY_AMENDMENT"
                and e.get("amendment_id") == AMENDMENT_ID]
    baseline = identity["budget_pre_amendment"]
    expected = {
        "event": "BUDGET_CATEGORY_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
        "amendment_id": AMENDMENT_ID, "card_id": CARD_ID, "source_category": "control_replay",
        "destination_category": "pilot", "fits": 4, "global_cap": 60,
        "original_caps": {"continuation": 24, "control_replay": 6, "pilot": 26, "technical_retry": 4},
        "effective_caps": {"continuation": 24, "control_replay": 2, "pilot": 30, "technical_retry": 4},
        "pre_amendment_ledger_sha256": baseline["ledger_sha256"],
        "pre_amendment_event_rows": baseline["event_rows"],
        "original_authorization_sha256": baseline["authorization_sha256"],
        "freeze_signature": signature, "source_commit": identity["source_commit"],
        "reason": "User-authorized four-fit A01 pilot transfer from unused control_replay capacity; no total-cap increase.",
    }
    if existing:
        if len(existing) != 1 or any(existing[0].get(k) != v for k, v in expected.items()):
            raise RuntimeError("Existing A01 budget amendment conflicts; refusing duplicate or rewrite")
        return existing[0]
    current = _budget_baseline_snapshot(require_original_state=False)
    if current != baseline:
        raise RuntimeError(f"Live ledger changed after A01 freeze and before amendment: {current}")
    if any(e.get("event") == "BUDGET_CATEGORY_AMENDMENT" for e in events):
        raise RuntimeError("Unexpected prior category amendment exists; refusing to stack budget changes")
    expected["at_utc"] = datetime.now(timezone.utc).isoformat()
    _append_jsonl(LEDGER, expected)
    saved = read_events(LEDGER)
    return _effective_budget(saved, require_reservation=False) | {"amendment_event": expected}


def reserve_a01() -> dict[str, Any]:
    identity, signature = _load_frozen()
    events = read_events(LEDGER)
    amendment = [e for e in events if e.get("event") == "BUDGET_CATEGORY_AMENDMENT"
                 and e.get("amendment_id") == AMENDMENT_ID]
    if len(amendment) != 1:
        raise RuntimeError("A01 amendment must be appended before the pilot reservation")
    if amendment[0].get("freeze_signature") != signature or amendment[0].get("source_commit") != identity["source_commit"]:
        raise RuntimeError("A01 budget amendment does not bind to the frozen code/configuration")
    existing = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                and e.get("reservation_key") == RESERVATION_KEY]
    if existing:
        if len(existing) != 1 or int(existing[0].get("fits", -1)) != 4 or existing[0].get("category") != "pilot" or existing[0].get("amendment_id") != AMENDMENT_ID:
            raise RuntimeError("Existing A01 reservation differs from exactly four pilot fits")
        return _effective_budget(events, require_reservation=True)
    current = _effective_budget(events, require_reservation=False)
    if current["active_reserved"] != 50 or current["active_by_category"].get("pilot") != 26:
        raise RuntimeError(f"Live budget changed before A01 reservation: {current}")
    event = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "amendment_id": AMENDMENT_ID,
        "freeze_signature": signature, "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(LEDGER, event)
    return _effective_budget(read_events(LEDGER), require_reservation=True)


def _fit_id(cutoff: int, arm: str) -> str:
    return f"{CARD_ID}_{cutoff}_{arm}"


def _fit_events(fit_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    events = read_events(LEDGER)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) > 1 or len(terminal) > 1:
        raise RuntimeError(f"Duplicate A01 ledger events for {fit_id}")
    return starts[0] if starts else None, terminal[0] if terminal else None


def _atomic_joblib(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        joblib.dump(value, stream, compress=3)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return _sha(path)


def _acquire_fit_lock() -> Path:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = RUN_ROOT / ".active_fit.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "at_utc": datetime.now(timezone.utc).isoformat()}))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"A01 training lock exists; inspect before proceeding: {lock}") from exc
    return lock


def _bundle_identity(identity: Mapping[str, Any], signature: str, cutoff: int, arm: str,
                     model_sha: str, state_sha: str) -> dict[str, Any]:
    return {"campaign": CARD_ID, "stage": STAGE, "cutoff": int(cutoff), "arm": arm,
            "source_commit": identity["source_commit"], "protocol_sha256": _sha(PROTOCOL_PATH),
            "freeze_signature": signature, "model_joblib_sha256": model_sha,
            "fit_state_sha256": state_sha, "threshold": THRESHOLD}


def _release_if_terminal() -> dict[str, Any] | None:
    events = read_events(LEDGER)
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("reservation_key") == RESERVATION_KEY]
    if not reservation:
        return None
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("reservation_key") == RESERVATION_KEY]
    terminals = {e.get("fit_id") for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    if any(e.get("fit_id") not in terminals for e in starts):
        return None
    return release_unused(LEDGER, card_id=CARD_ID, stage=STAGE)


def fit_one(cutoff: int, arm: str) -> dict[str, Any]:
    if cutoff not in ORIGINS or arm not in ARMS:
        raise ValueError(f"Allowed cutoff/arm pairs: {FIT_ORDER}")
    identity, signature = _load_frozen()
    if _effective_budget(read_events(LEDGER), require_reservation=True)["active_reserved"] != 54:
        raise RuntimeError("A01 effective active budget is not the reviewed 54/60 state")
    position = FIT_ORDER.index((cutoff, arm))
    for previous_cutoff, previous_arm in FIT_ORDER[:position]:
        _start, terminal = _fit_events(_fit_id(previous_cutoff, previous_arm))
        if terminal is None:
            raise RuntimeError(f"Frozen fit order requires {_fit_id(previous_cutoff, previous_arm)} terminal first")
    train_path, frame, target, cycles = _load_train()
    controls = _read_c16_controls(frame, cycles)[cutoff]
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
    score_rows = controls["rows"]
    plan = recency_weight_plan(cycles[fit_rows], target[fit_rows], cutoff)
    weights = plan["recency_weights"] if arm == "RECENCY" else plan["shuffled_weights"]
    if plan["diagnostics"] != identity["recipe"]["weights"]["diagnostics_by_cutoff"][str(cutoff)]:
        raise RuntimeError(f"A01 weight/permutation diagnostics changed from freeze at cutoff {cutoff}")
    fit_frame = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    score_frame = frame.iloc[score_rows].loc[:, list(FEATURE_NAMES)].copy()
    memory = _memory_estimate(fit_frame)
    if memory["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_MEMORY before A01 fit at cutoff {cutoff}: {memory}")
    if (len(fit_rows) != int(identity["recipe"]["fit_rows_expected"][str(cutoff)])
            or len(score_rows) != int(identity["recipe"]["score_rows_expected"][str(cutoff)])):
        raise RuntimeError("A01 frozen fit/score row denominators changed")
    fit_id = _fit_id(cutoff, arm)
    fit_dir = RUN_ROOT / f"fit{cutoff}_{arm.lower()}"
    model_path, state_path, bundle_path = fit_dir / "fitted_state.joblib", fit_dir / "fit_state.json", fit_dir / "bundle"
    start, terminal = _fit_events(fit_id)
    if terminal and terminal.get("event") == "FIT_FAILED":
        raise RuntimeError(f"A01 fit {fit_id} already failed; implicit retries are prohibited")
    lock = _acquire_fit_lock()
    try:
        fit_dir.mkdir(parents=True, exist_ok=True)
        if not model_path.is_file():
            if start is not None:
                finish_fit(LEDGER, fit_id=fit_id, status="FIT_FAILED",
                           error="Interrupted before durable fitted model; no retry/refit permitted")
                raise RuntimeError(f"A01 started fit {fit_id} has no durable model; recorded consumed failure")
            start_fit(LEDGER, fit_id=fit_id, card_id=CARD_ID, stage=STAGE,
                      cutoff=cutoff, component=arm)
            estimator = RecencySparseClassifier(cutoff, arm)
            warnings_seen: list[str] = []
            started_at = perf_counter()
            try:
                with warnings.catch_warnings(record=True) as found, threadpool_limits(limits=THREADS):
                    warnings.simplefilter("always")
                    estimator.fit(fit_frame, target[fit_rows], weights)
                    warnings_seen = [f"{w.category.__name__}: {w.message}" for w in found]
                fit_seconds = perf_counter() - started_at
                model_sha = _atomic_joblib({"model": estimator, "fit_seconds": fit_seconds,
                                            "warnings": warnings_seen}, model_path)
                model_state = estimator.audit_state()
                state = {
                    "campaign": CARD_ID, "fit_id": fit_id, "cutoff": int(cutoff), "arm": arm,
                    "source_commit": identity["source_commit"], "freeze_signature": signature,
                    "protocol_sha256": _sha(PROTOCOL_PATH), "official_train_sha256": _sha(train_path),
                    "fit_rows_sha256": _sha_int64(fit_rows), "score_rows_sha256": _sha_int64(score_rows),
                    "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
                    "fit_labels_passed_to_estimator": True, "score_labels_passed_to_fit": False,
                    "test_or_sample_data_used": False, "kaggle_access": False,
                    "sample_weight_diagnostics": plan["diagnostics"],
                    "model_audit_state": model_state, "model_joblib_sha256": model_sha,
                    "fit_seconds": float(fit_seconds), "warnings": warnings_seen,
                    "resource_preflight": memory,
                }
                state["fit_state_sha256"] = canonical_sha256(state)
                _write_once(state_path, state)
            except Exception as exc:
                _start, existing_terminal = _fit_events(fit_id)
                if _start and existing_terminal is None:
                    finish_fit(LEDGER, fit_id=fit_id, status="FIT_FAILED",
                               error=f"{type(exc).__name__}: {exc}")
                raise
        else:
            if start is None:
                raise RuntimeError(f"A01 durable model exists without FIT_STARTED for {fit_id}")
            stored = joblib.load(model_path)
            estimator = stored["model"]
            model_sha = _sha(model_path)
            if not state_path.is_file():
                raise RuntimeError(f"A01 model persisted but fit-state receipt is missing: {fit_id}")
            state = _json(state_path)
            saved_state_hash = state.get("fit_state_sha256")
            state_without_hash = {k: v for k, v in state.items() if k != "fit_state_sha256"}
            if (saved_state_hash != canonical_sha256(state_without_hash)
                    or state.get("model_joblib_sha256") != model_sha
                    or state.get("freeze_signature") != signature
                    or state.get("fit_id") != fit_id
                    or estimator.audit_state() != state.get("model_audit_state")):
                raise RuntimeError(f"A01 durable model/state identity mismatch for {fit_id}")
        if terminal and terminal.get("event") == "FIT_SAVED":
            if not bundle_path.is_dir():
                raise RuntimeError(f"Ledger claims A01 fit saved but bundle is missing: {fit_id}")
            bundle_id = _bundle_identity(identity, signature, cutoff, arm, model_sha, _sha(state_path))
            saved = read_research_bundle(bundle_path, bundle_id, task="classification")
            return {"status": "RESUMED_SAVED", "fit_id": fit_id,
                    "bundle_receipt_sha256": saved.receipt["receipt_sha256"]}
        if terminal is not None and terminal.get("event") != "FIT_SAVED":
            raise RuntimeError(f"Unexpected terminal A01 event for {fit_id}: {terminal}")
        if bundle_path.exists():
            bundle_id = _bundle_identity(identity, signature, cutoff, arm, model_sha, _sha(state_path))
            saved = read_research_bundle(bundle_path, bundle_id, task="classification")
            receipt = saved.receipt
        else:
            raw = estimator.predict_probability(score_frame)
            route = route_probabilities(raw if arm == "RECENCY" else np.zeros_like(raw),
                                        raw if arm == "SHUFFLED_WITHIN_CLASS" else np.zeros_like(raw),
                                        controls["sparse25_raw"], controls["c2b"], controls["cb"], controls["missing"])
            prediction = route["recency_candidate"] if arm == "RECENCY" else route["shuffled_weight_control"]
            ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
            payload = {
                "score_rows": score_rows.astype(np.int64), "score_record_ids": ids,
                "cycle_code": cycles[score_rows].astype(np.int16), "prediction": prediction.astype(np.float64),
                "baseline_prediction": route["sparse25_baseline"].astype(np.float64),
                "raw_probability": raw.astype(np.float64), "sparse25_raw_probability": controls["sparse25_raw"],
                "c2b_probability": controls["c2b"], "cb_probability": controls["cb"],
                "stream_missing": controls["missing"].astype(bool),
                "payload_meta": {"campaign": CARD_ID, "stage": STAGE, "cutoff": int(cutoff),
                                 "arm": arm, "threshold": THRESHOLD,
                                 "probability_semantics": "routed probability; fixed threshold 0.50",
                                 "candidate_policy": "RECENCY" if arm == "RECENCY" else "matched shuffled-weight control",
                                 "score_labels_passed_to_fit": False, "test_read": False,
                                 "sample_submission_read": False, "kaggle_access": False,
                                 "model_joblib_sha256": model_sha, "fit_state_sha256": _sha(state_path)},
            }
            bundle_id = _bundle_identity(identity, signature, cutoff, arm, model_sha, _sha(state_path))
            receipt = publish_bundle(bundle_path, payload, bundle_id, task="classification")
            saved = read_research_bundle(bundle_path, bundle_id, task="classification")
            if saved.receipt["receipt_sha256"] != receipt["receipt_sha256"]:
                raise RuntimeError(f"A01 bundle failed readback verification for {fit_id}")
        finish_fit(LEDGER, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=receipt["receipt_sha256"])
        return {"status": "FIT_SAVED", "fit_id": fit_id, "cutoff": cutoff, "arm": arm,
                "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
                "fit_seconds": float(state["fit_seconds"]), "memory": memory,
                "model_joblib_sha256": model_sha, "fit_state_sha256": _sha(state_path),
                "bundle_receipt_sha256": receipt["receipt_sha256"]}
    finally:
        lock.unlink(missing_ok=True)


def view_definitions() -> dict[str, list[tuple[int, int, int]]]:
    return VIEWS


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, (0, 1)).all():
        raise ValueError("A01 metrics need aligned, nonempty binary labels")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("A01 probabilities must be finite and in [0,1]")
    pred = (p >= THRESHOLD).astype(np.int8)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "positive_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "negative_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "positive_rate": float(y.mean()), "predicted_positive_rate": float(pred.mean()),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def _origin_bins(frame: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, list[float]]:
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    fit = (cycles >= 47) & (cycles <= cutoff)
    finite = parsed[fit & np.isfinite(parsed)]
    if not len(finite):
        return np.full(len(frame), -1, dtype=np.int64), []
    edges = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
    labels = np.full(len(frame), -1, dtype=np.int64)
    valid = np.isfinite(parsed)
    labels[valid] = np.searchsorted(edges, parsed[valid], side="right")
    return labels, edges.astype(float).tolist()


def _load_origin_outputs(identity: Mapping[str, Any], signature: str, cutoff: int,
                         frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for arm in ARMS:
        fit_dir = RUN_ROOT / f"fit{cutoff}_{arm.lower()}"
        model_path, state_path, bundle_path = fit_dir / "fitted_state.joblib", fit_dir / "fit_state.json", fit_dir / "bundle"
        fit_id = _fit_id(cutoff, arm)
        starts, terminal = _fit_events(fit_id)
        if not starts or not terminal or terminal.get("event") != "FIT_SAVED":
            raise RuntimeError(f"A01 model fit is not successfully terminal: {fit_id}")
        if not model_path.is_file() or not state_path.is_file():
            raise RuntimeError(f"A01 fitted model/state is missing: {fit_id}")
        model_sha = _sha(model_path)
        state = _json(state_path)
        if state.get("model_joblib_sha256") != model_sha:
            raise RuntimeError(f"A01 model hash differs from state receipt: {fit_id}")
        bundle_id = _bundle_identity(identity, signature, cutoff, arm, model_sha, _sha(state_path))
        bundle = read_research_bundle(bundle_path, bundle_id, task="classification")
        arrays = bundle.arrays
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        ids = frame.iloc[expected_rows]["record_id"].astype(str).to_numpy(dtype=str)
        if (not np.array_equal(arrays["score_rows"], expected_rows)
                or not np.array_equal(arrays["score_record_ids"].astype(str), ids)
                or not np.array_equal(arrays["cycle_code"].astype(np.int64), cycles[expected_rows])
                or not np.array_equal(arrays["stream_missing"].astype(bool), stream_missing_mask(frame.iloc[expected_rows]))
                or terminal.get("receipt_sha256") != bundle.receipt.get("receipt_sha256")):
            raise RuntimeError(f"A01 saved bundle row/order/route identity failed: {fit_id}")
        outputs[arm] = {key: np.asarray(value) for key, value in arrays.items()}
        outputs[arm]["fit_state"] = state
        outputs[arm]["receipt"] = bundle.receipt
    first, second = outputs[ARMS[0]], outputs[ARMS[1]]
    for key in ("score_rows", "score_record_ids", "cycle_code", "baseline_prediction",
                "sparse25_raw_probability", "c2b_probability", "cb_probability", "stream_missing"):
        if not np.array_equal(first[key], second[key]):
            raise RuntimeError(f"A01 arm score/control arrays differ at cutoff {cutoff}: {key}")
    route = route_probabilities(first["raw_probability"], second["raw_probability"],
                                first["sparse25_raw_probability"], first["c2b_probability"],
                                first["cb_probability"], first["stream_missing"])
    np.testing.assert_allclose(route["sparse25_baseline"], first["baseline_prediction"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(route["recency_candidate"], first["prediction"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(route["shuffled_weight_control"], second["prediction"], rtol=0, atol=1e-12)
    return {"score_rows": first["score_rows"].astype(np.int64),
            "score_record_ids": first["score_record_ids"].astype(str),
            "cycle_code": first["cycle_code"].astype(np.int64),
            "stream_missing": first["stream_missing"].astype(bool),
            "baseline_prediction": route["sparse25_baseline"],
            "prediction": route["recency_candidate"],
            "matched_prediction": route["shuffled_weight_control"],
            "c2b_diagnostic": np.where(first["stream_missing"].astype(bool),
                                       first["cb_probability"], first["c2b_probability"]),
            "origin": np.full(len(first["score_rows"]), cutoff, dtype=np.int64),
            "fit_states": {arm: outputs[arm]["fit_state"] for arm in ARMS},
            "receipts": {arm: outputs[arm]["receipt"] for arm in ARMS}}


def _collect_view(outputs: Mapping[int, Mapping[str, Any]], blocks: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    parts = []
    keys = ("score_rows", "prediction", "baseline_prediction", "matched_prediction",
            "c2b_diagnostic", "origin")
    for cutoff, low, high in blocks:
        record = outputs[cutoff]
        take = (record["cycle_code"] >= low) & (record["cycle_code"] <= high)
        parts.append({key: np.asarray(record[key])[take] for key in keys})
    joined = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.argsort(joined["score_rows"], kind="stable")
    joined = {key: value[order] for key, value in joined.items()}
    if len(np.unique(joined["score_rows"])) != len(joined["score_rows"]):
        raise RuntimeError("A01 evaluation view contains duplicate official Train rows")
    return joined


def evaluate_views(outputs: Mapping[int, Mapping[str, Any]], frame: pd.DataFrame,
                   target: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    cycles_all = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    bins_by_origin = {cutoff: _origin_bins(frame, cutoff)[0] for cutoff in ORIGINS}
    metrics_by_view: dict[str, Any] = {}
    groups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    for name, blocks in VIEWS.items():
        data = _collect_view(outputs, blocks)
        rows = data["score_rows"]
        y = np.asarray(target, dtype=np.int8)[rows]
        probs = {"recency": data["prediction"], "sparse25": data["baseline_prediction"],
                 "shuffled": data["matched_prediction"], "C2_B_diagnostic": data["c2b_diagnostic"]}
        m = {key: _metric(y, probability) for key, probability in probs.items()}
        gains = {key: {view_metric: (None if m["recency"][view_metric] is None or m[key][view_metric] is None
                                     else float(m["recency"][view_metric] - m[key][view_metric]))
                       for view_metric in ("macro_f1", "positive_recall", "positive_precision",
                                           "predicted_positive_rate", "roc_auc", "average_precision", "log_loss")}
                  for key in ("sparse25", "shuffled", "C2_B_diagnostic")}
        metrics_by_view[name] = {"n": int(len(rows)), "models": m, "gains": gains,
                                 "origins": sorted(set(data["origin"].tolist())),
                                 "score_rows_sha256": _sha_int64(rows)}
        for cycle in sorted(np.unique(cycles_all[rows]).tolist()):
            take = cycles_all[rows] == cycle
            per_cycle.append({"view": name, "cycle": int(cycle), "n": int(take.sum()),
                              **{f"{key}_macro_f1": _metric(y[take], probability[take])["macro_f1"]
                                 for key, probability in probs.items()}})
        row_frame = frame.iloc[rows]
        missing = stream_missing_mask(row_frame).astype(bool)
        origins = data["origin"].astype(np.int64)
        quartiles = np.asarray([bins_by_origin[int(origin)][int(row)] for row, origin in zip(rows, origins)])
        groupings = {
            "fit_defined_text_load_quartile": np.asarray(["missing" if value < 0 else f"Q{min(int(value)+1,4)}"
                                                          for value in quartiles]),
            "stream_availability": np.where(missing, "missing", "present"),
            "entry_friction": row_frame["entry_friction"].astype(str).to_numpy(),
        }
        for field, labels in groupings.items():
            for label in sorted(np.unique(labels).tolist()):
                take = labels == label
                gy = y[take]
                n0, n1 = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                supported = len(gy) >= SUBGROUP_SUPPORT["minimum_n"] and min(n0, n1) >= SUBGROUP_SUPPORT["minimum_each_class_n"]
                candidate_metric = _metric(gy, probs["recency"][take])
                baseline_metric = _metric(gy, probs["sparse25"][take])
                matched_metric = _metric(gy, probs["shuffled"][take])
                f1_harm = float(baseline_metric["macro_f1"] - candidate_metric["macro_f1"])
                recall_harm = float(baseline_metric["positive_recall"] - candidate_metric["positive_recall"])
                groups.append({"view": name, "field": field, "group": str(label), "n": int(len(gy)),
                               "class_0_n": n0, "class_1_n": n1, "supported": bool(supported),
                               "recency_macro_f1": candidate_metric["macro_f1"],
                               "sparse25_macro_f1": baseline_metric["macro_f1"],
                               "shuffled_macro_f1": matched_metric["macro_f1"],
                               "f1_harm_vs_sparse25": f1_harm,
                               "recency_positive_recall": candidate_metric["positive_recall"],
                               "sparse25_positive_recall": baseline_metric["positive_recall"],
                               "positive_recall_harm_vs_sparse25": recall_harm,
                               "fails_legacy_f1_guard": bool(supported and f1_harm > SUBGROUP_GUARDS["macro_f1_harm_max"]),
                               "fails_legacy_recall_guard": bool(supported and recall_harm > SUBGROUP_GUARDS["positive_recall_harm_max"]),
                               "risk_review_hold": bool(supported and (f1_harm > RISK_REVIEW["macro_f1_harm_gt"]
                                                                         or recall_harm > RISK_REVIEW["positive_recall_harm_gt"]))})
    g = metrics_by_view
    gains = {name: float(g[name]["gains"]["sparse25"]["macro_f1"])
             for name in ("primary", "W1", "W4", "H12_55", "H12_67", "H12_pooled")}
    h12_vs_shuffled = float(g["H12_pooled"]["gains"]["shuffled"]["macro_f1"])
    checks = {
        "primary_nonnegative_vs_sparse25": gains["primary"] >= GATES["primary_gain_vs_sparse25_min"],
        "primary_nonnegative_vs_shuffled": g["primary"]["gains"]["shuffled"]["macro_f1"] >= GATES["primary_gain_vs_shuffled_min"],
        "H12_min_gain_vs_sparse25": gains["H12_pooled"] >= GATES["H12_gain_vs_sparse25_min"],
        "H12_min_gain_vs_shuffled": h12_vs_shuffled >= GATES["H12_gain_vs_shuffled_min"],
        "H12_55_harm_guard": gains["H12_55"] >= -GATES["H12_55_max_harm_vs_sparse25"],
        "H12_67_harm_guard": gains["H12_67"] >= -GATES["H12_67_max_harm_vs_sparse25"],
        "W1_harm_guard": gains["W1"] >= -GATES["W1_max_harm_vs_sparse25"],
        "W4_harm_guard": gains["W4"] >= -GATES["W4_max_harm_vs_sparse25"],
        "protected_legacy_subgroup_guards": not any(row["supported"] and
            (row["fails_legacy_f1_guard"] or row["fails_legacy_recall_guard"]) for row in groups),
    }
    risk_groups = [row for row in groups if row["risk_review_hold"]]
    gate = {"status": "PASS" if all(checks.values()) else "NO_GO_GATE_FAILED",
            "checks": checks, "gains_vs_sparse25": gains,
            "H12_pooled_gain_vs_shuffled": h12_vs_shuffled,
            "thresholds": {"frozen_gates": GATES, "subgroup_support": SUBGROUP_SUPPORT,
                           "protected_legacy_guards": SUBGROUP_GUARDS, "risk_review_hold": RISK_REVIEW},
            "supported_subgroup_count": sum(row["supported"] for row in groups),
            "supported_subgroup_failures": [row for row in groups if row["supported"] and
                                            (row["fails_legacy_f1_guard"] or row["fails_legacy_recall_guard"])],
            "risk_review_hold_groups": risk_groups,
            "scientific_verdict": ("RISK_REVIEW_HOLD" if all(checks.values()) and risk_groups else
                                   "QUALIFIED_FOR_REVIEW" if all(checks.values()) else "NO_GO_GATE_FAILED")}
    return metrics_by_view, groups, per_cycle, gate


def summarize() -> dict[str, Any]:
    identity, signature = _load_frozen()
    if (RUN_ROOT / ".active_fit.lock").exists():
        raise RuntimeError("Cannot summarize while an A01 fit lock is active")
    train_path, frame, target, cycles = _load_train()
    fit_states: dict[str, Any] = {}
    outputs: dict[int, dict[str, Any]] = {}
    for cutoff, arm in FIT_ORDER:
        fit_id = _fit_id(cutoff, arm)
        start, terminal = _fit_events(fit_id)
        fit_states[fit_id] = {"status": "NOT_STARTED" if not start else
                              "PENDING" if not terminal else terminal["event"],
                              "terminal": terminal}
    if not all(value["status"] == "FIT_SAVED" for value in fit_states.values()):
        return {"status": "INCOMPLETE_CHECKPOINT", "campaign": CARD_ID,
                "fit_status": fit_states, "official_fits_planned": 4,
                "test_read": False, "sample_submission_read": False, "kaggle_access": False}
    outputs = {cutoff: _load_origin_outputs(identity, signature, cutoff, frame, cycles)
               for cutoff in ORIGINS}
    metrics, groups, per_cycle, gate = evaluate_views(outputs, frame, target)
    release_event = _release_if_terminal()
    events = read_events(LEDGER)
    accounting = _effective_budget(events, require_reservation=True)
    budget = ledger_summary(LEDGER)
    if budget["started_and_consumed"] != 54 or budget["saved"] != 52 or budget["failed_or_interrupted_started"] != 2:
        raise RuntimeError(f"A01 final 54-started budget reconciliation unexpected: {budget}")
    result = {
        "status": "COMPLETE_REVIEW_REQUIRED", "campaign": CARD_ID,
        "scientific_verdict": gate["scientific_verdict"], "freeze_signature": signature,
        "source_commit": identity["source_commit"], "code_sha256": identity["code_sha256"],
        "protocol_sha256": _sha(PROTOCOL_PATH), "official_train_sha256": _sha(train_path),
        "official_train_rows": int(len(frame)), "official_estimator_fits": 4,
        "fit_status": fit_states, "metrics_by_view": metrics,
        "protected_groups": groups, "per_cycle": per_cycle, "frozen_gate": gate,
        "fit_audits": {str(cutoff): {arm: outputs[cutoff]["fit_states"][arm]
                                     for arm in ARMS} for cutoff in ORIGINS},
        "bundle_receipts": {str(cutoff): {arm: outputs[cutoff]["receipts"][arm]
                                           for arm in ARMS} for cutoff in ORIGINS},
        "budget_amendment": {"amendment_id": AMENDMENT_ID, "source": "control_replay",
                             "destination": "pilot", "fits": 4,
                             "original_caps": {"continuation": 24, "control_replay": 6,
                                               "pilot": 26, "technical_retry": 4},
                             "effective_caps": accounting["category_caps"], "global_cap": 60},
        "budget": {"ledger_summary_original": budget, "effective_amended_summary": accounting,
                   "release_event": release_event, "ledger_sha256": _sha(LEDGER),
                   "ledger_event_rows": len(events)},
        "interpretation_limits": [
            "Frozen two-origin pilot; not an untouched holdout evaluation.",
            "The recency arm is compared with frozen sparse25 and a within-class shuffled-weight control.",
            "The shuffled control preserves per-class weight masses and the same estimator recipe; it is not an independent model family.",
            "Subgroup findings are descriptive guards on supported cells and do not establish causal effects.",
            "No Test, SampleSubmission, full-Train refit, Kaggle access, or upload occurred.",
        ],
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
    }
    _write_once(RESULTS_PATH, result)
    _write_csv_once(REPORT_ROOT / "protected_groups.csv", groups)
    _write_csv_once(REPORT_ROOT / "per_cycle.csv", per_cycle)
    return result


def _write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        expected = [{key: str(row.get(key, "")) for key in columns} for row in rows]
        if previous != expected:
            raise RuntimeError(f"Immutable A01 CSV differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("amend-budget")
    sub.add_parser("reserve")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("cutoff", type=int, choices=ORIGINS)
    fit_parser.add_argument("arm", choices=ARMS)
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "amend-budget":
        result = append_amendment()
    elif args.command == "reserve":
        result = reserve_a01()
    elif args.command == "fit":
        result = fit_one(args.cutoff, args.arm)
    else:
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
