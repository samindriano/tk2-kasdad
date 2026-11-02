"""Recover ClassOnly CSV/manifest from the already saved retry bundle; zero estimator fits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_measured_classification as measured  # noqa: E402
from scripts import optimization_v3_measured_classification_retry as retry  # noqa: E402
from scripts import prepare_c16_r22_fulltrain_submissions as fulltrain  # noqa: E402
from src.models.research_next_budget import ledger_summary, read_events, _append  # noqa: E402
from src.models.research_next_contract import (  # noqa: E402
    canonical_sha256, file_sha256, read_bundle,
)
from src.utils.paths import task_files  # noqa: E402

RECOVERY_EVENT = "OUTPUT_PREPARATION_RECOVERED"
REPORT_PATH = measured.REPORT_ROOT / "CLASS_ONLY_RETRY_PREPARATION.json"
RECOVERY_MANIFEST_PATH = measured.REPORT_ROOT / "CLASS_ONLY_SUBMISSION_MANIFEST.json"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _verify_frozen_lineage() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    amendment = json.loads(retry.AMENDMENT_FREEZE.read_text(encoding="utf-8"))
    identity = amendment.get("identity", {})
    _require(amendment.get("freeze_signature") == canonical_sha256(identity),
             "technical retry amendment signature failed")
    parent_context = measured._context()
    parent = measured._verify_freeze(parent_context)
    _require(parent.get("freeze_signature") == identity.get("parent_freeze_signature") and
             file_sha256(measured.FREEZE_PATH) == identity.get("parent_freeze_sha256"),
             "original measured freeze changed")
    for path_name, expected in identity["retry_source_sha256"].items():
        _require(file_sha256(ROOT / path_name) == expected,
                 f"prefit technical retry source changed: {path_name}")
    _require(file_sha256(measured.PROTOCOL) == identity["protocol_sha256"] and
             file_sha256(measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json") ==
             identity["temporal_result_sha256"] and
             file_sha256(retry.SYNTHETIC_REPORT) == identity["synthetic_preflight_sha256"],
             "technical retry frozen evidence changed")
    temporal = json.loads((measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json").read_text(encoding="utf-8"))
    _require(temporal.get("decision") == "BENCHMARK_GATES_FAIL" and
             temporal.get("gates", {}).get("primary_gain_vs_sparse25_ge_0_0002") is False,
             "N02 gate verdict changed")
    events = read_events(retry.LEDGER)
    original_failure = [e for e in events if e.get("fit_id") == "MEASURED_CLASS_ONLY_FULLTRAIN" and
                        e.get("event") == "FIT_FAILED"]
    retry_saved = [e for e in events if e.get("fit_id") == retry.RETRY_ID and e.get("event") == "FIT_SAVED"]
    _require(len(original_failure) == 1 and len(retry_saved) == 1,
             "original failure or saved technical retry is absent from append-only ledger")
    return amendment, temporal, events


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = task_files("classification")
    train_path, test_path, sample_path = paths["train"], paths["test"], paths["sample"]
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False, usecols=lambda column: column != "artifact_signal")
    sample = pd.read_csv(sample_path, low_memory=False)
    _require(len(train) == 50_608 and len(test) == 29_421 and len(sample) == 29_421,
             "official Classification rows changed")
    _require("artifact_signal" not in test.columns and
             list(sample.columns) == ["record_id", "artifact_signal"] and
             test.record_id.astype(str).equals(sample.record_id.astype(str)),
             "Test target boundary, Sample schema, or official ID order failed")
    hashes = {"train": file_sha256(train_path), "test": file_sha256(test_path),
              "sample": file_sha256(sample_path)}
    amendment = json.loads(retry.AMENDMENT_FREEZE.read_text(encoding="utf-8"))["identity"]
    _require(hashes == amendment["official_input_sha256"], "official inference input hashes changed")
    return train, test, sample, hashes


def _persisted_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    reuse_c2b, reuse_cb, reuse_missing, reuse_audit = fulltrain._verify_classification_reuse(train, test)
    with np.load(fulltrain.CLASS_CACHE, allow_pickle=False) as cache:
        c2b = np.asarray(cache["base"], dtype=np.float64)
        cb = np.asarray(cache["missing"], dtype=np.float64)
        missing = np.asarray(cache["missing_mask"], dtype=bool)
        baseline = np.asarray(cache["prediction"], dtype=np.float64)
    cache_manifest = json.loads(fulltrain.CLASS_MANIFEST.read_text(encoding="utf-8"))
    _require(np.array_equal(c2b, reuse_c2b) and np.array_equal(cb, reuse_cb) and
             np.array_equal(missing, reuse_missing) and
             measured._array_sha(baseline) == cache_manifest.get("prediction_sha256"),
             "attested reusable controls differ from the frozen full-Train cache")
    bundle_identity = {
        "batch_id": retry.TASK, "card_id": retry.RETRY_CARD, "stage": retry.RETRY_STAGE,
        "parent_freeze_signature": json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))["freeze_signature"],
        "technical_retry_amendment": "FULLTRAIN_RETRY_AMENDMENT_01",
        "candidate": "CLASS_ONLY", "cutoff": 79,
    }
    bundle = read_bundle(retry.RETRY_BUNDLE, bundle_identity, task="classification")
    saved = [e for e in read_events(retry.LEDGER) if e.get("fit_id") == retry.RETRY_ID and
             e.get("event") == "FIT_SAVED"]
    _require(len(saved) == 1 and saved[0].get("receipt_sha256") == bundle.receipt.get("receipt_sha256"),
             "saved fit ledger receipt does not match persisted model output")
    metadata = bundle.receipt["metadata"]["metadata"]
    state = metadata["fit_state"]
    _require(state.get("fit_rows") == 50_608 and state.get("fit_cycle_min") == 47 and
             state.get("fit_cycle_max") == 79 and state.get("test_data_used") is False and
             state.get("validation_labels_passed_to_fit") is False and
             metadata.get("test_target_column_read") is False and metadata.get("test_labels_used") is False,
             "saved ClassOnly fitted-state boundary failed")
    expected = np.where(missing, cb, 0.75 * c2b + 0.25 * bundle.arrays["raw_prediction"])
    _require(np.allclose(bundle.arrays["prediction"], expected, rtol=0, atol=1e-12) and
             np.array_equal(bundle.arrays["baseline_prediction"], baseline) and
             np.array_equal(bundle.arrays["label"], (expected >= 0.50).astype(np.int8)),
             "saved retry output does not reconstruct the frozen route/threshold")
    return bundle.arrays, {"bundle": bundle, "metadata": metadata}, reuse_audit


def _record_recovery_event(csv_sha: str) -> dict[str, Any]:
    events = read_events(retry.LEDGER)
    existing = [event for event in events if event.get("event") == RECOVERY_EVENT and
                event.get("fit_id") == retry.RETRY_ID]
    expected = {
        "event": RECOVERY_EVENT, "batch_id": retry.TASK, "fit_id": retry.RETRY_ID,
        "failed_csv_writer": "optimization_v3_measured_classification_retry._prepare_csv_and_manifest",
        "recovery_action": "rebuild CSV/manifest from FIT_SAVED bundle; no estimator fit",
        "csv_sha256": csv_sha, "official_fit_consumed": False, "kaggle_called": False,
    }
    if existing:
        _require(len(existing) == 1 and all(existing[0].get(key) == value for key, value in expected.items()),
                 "existing output recovery ledger event differs")
        return existing[0]
    return _append(retry.LEDGER, expected)


def recover_outputs() -> dict[str, Any]:
    amendment, temporal, _ = _verify_frozen_lineage()
    train, test, sample, hashes = _load_inputs()
    arrays, fit_info, reuse_audit = _persisted_predictions(train, test)
    frame = measured._submission_frame(test, sample, arrays["label"], "artifact_signal")
    if not retry.CSV_PATH.exists():
        csv_payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
        fulltrain._write_once(retry.CSV_PATH, csv_payload)
    saved_csv = pd.read_csv(retry.CSV_PATH)
    _require(list(saved_csv.columns) == list(frame.columns) == ["record_id", "artifact_signal"] and
             len(saved_csv) == 29_421 and
             saved_csv.record_id.astype(str).equals(frame.record_id.astype(str)) and
             not saved_csv.record_id.astype(str).duplicated().any() and
             np.array_equal(saved_csv.artifact_signal.to_numpy(dtype=np.int64),
                            frame.artifact_signal.to_numpy(dtype=np.int64)) and
             set(saved_csv.artifact_signal.unique()) <= {0, 1},
             "ClassOnly CSV ID/value/schema QA failed")
    csv_info = {"path_gitignored": retry.CSV_PATH.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(retry.CSV_PATH), "rows": len(frame),
                "columns": list(frame.columns),
                "ordered_ids_sha256": hashlib.sha256(("\n".join(frame.record_id.astype(str)) + "\n").encode()).hexdigest()}
    csv_sha = csv_info["sha256"]
    recovery_event = _record_recovery_event(csv_sha)
    inputs = {**hashes, "original_measured_freeze": file_sha256(measured.FREEZE_PATH),
              "measured_protocol": file_sha256(measured.PROTOCOL),
              "technical_retry_amendment_freeze": file_sha256(retry.AMENDMENT_FREEZE)}
    manifest = {
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED", "task": "classification",
        "candidate": "CLASS_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": measured._head(), "parent_freeze_signature": amendment["identity"]["parent_freeze_signature"],
        "technical_retry_amendment_signature": amendment["freeze_signature"],
        "technical_retry_of": "MEASURED_CLASS_ONLY_FULLTRAIN",
        "recipe": "N01 exact class_only sparse recipe; all official Train cycles 47-79; present .75 C2_B + .25 weighted sparse expert; missing frozen CB_MISSING; threshold 0.50",
        "official_input_sha256": inputs, "fit_rows": 50_608, "fit_cycle_range": [47, 79],
        "test_rows": 29_421, "fit_state_sha256": fit_info["metadata"]["fit_state"]["state_sha256"],
        "weight_plan": fit_info["metadata"]["weight_plan_diagnostics"],
        "fit_bundle": retry.RETRY_BUNDLE.relative_to(ROOT).as_posix(),
        "fit_receipt_sha256": fit_info["bundle"].receipt["receipt_sha256"],
        "fit_ledger": ledger_summary(retry.LEDGER), "reuse_audit": reuse_audit,
        "test_target_column_read": False, "test_labels_used": False,
        "preprocessing_fit_only_train": True,
        "csv": {**csv_info, "labels": {str(int(k)): int(v) for k, v in
                                       zip(*np.unique(arrays["label"], return_counts=True))}},
        "output_recovery": {"event": RECOVERY_EVENT, "reason": "post-FIT_SAVED CSV writer type mismatch",
                             "estimator_refit": False, "ledger_event_sha256": hashlib.sha256(
                                 json.dumps(recovery_event, sort_keys=True, allow_nan=False).encode()).hexdigest()},
        "kaggle_called": False, "submission_ref": None,
    }
    if RECOVERY_MANIFEST_PATH.exists():
        saved_manifest = json.loads(RECOVERY_MANIFEST_PATH.read_text(encoding="utf-8"))
        _require(saved_manifest.get("fit_receipt_sha256") == manifest["fit_receipt_sha256"] and
                 saved_manifest.get("csv", {}).get("sha256") == csv_sha,
                 "existing ClassOnly manifest differs; refusing overwrite")
        manifest = saved_manifest
    else:
        measured._write_once(RECOVERY_MANIFEST_PATH, manifest)
    result = {
        "status": "CLASS_ONLY_CSV_PREPARED_VALIDATED_NOT_SUBMITTED",
        "candidate": "CLASS_ONLY", "csv": csv_info,
        "manifest_path": RECOVERY_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": file_sha256(RECOVERY_MANIFEST_PATH),
        "fit_bundle": retry.RETRY_BUNDLE.relative_to(ROOT).as_posix(),
        "fit_receipt_sha256": fit_info["bundle"].receipt["receipt_sha256"],
        "original_fit_failed_preserved": True,
        "technical_retry_fit_saved": True, "estimator_refit_during_recovery": False,
        "n02_status": "NO_CSV_BENCHMARK_GATES_FAILED" if temporal["decision"] == "BENCHMARK_GATES_FAIL" else temporal["decision"],
        "test_labels_used": False, "kaggle_called": False,
        "fit_ledger": ledger_summary(retry.LEDGER),
    }
    if REPORT_PATH.exists():
        existing = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        _require(existing.get("fit_receipt_sha256") == result["fit_receipt_sha256"] and
                 existing.get("csv", {}).get("sha256") == csv_sha,
                 "existing recovery report differs; refusing overwrite")
        return existing
    measured._write_once(REPORT_PATH, result)
    return result


if __name__ == "__main__":
    print(json.dumps(recover_outputs(), sort_keys=True))
