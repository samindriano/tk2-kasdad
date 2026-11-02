"""Independent zero-fit audit after ClassOnly bundle and CSV recovery."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_measured_classification as measured  # noqa: E402
from scripts import optimization_v3_measured_classification_retry as retry  # noqa: E402
from scripts import prepare_c16_r22_fulltrain_submissions as fulltrain  # noqa: E402
from src.models.research_next_budget import ledger_summary, read_events  # noqa: E402
from src.models.research_next_contract import file_sha256, read_bundle  # noqa: E402
from src.models.research_next_n01 import FEATURE_NAMES  # noqa: E402
from src.models.round1_models import _fit_fingerprint  # noqa: E402
from src.utils.paths import task_files  # noqa: E402

OUTPUT = measured.REPORT_ROOT / "CLASS_ONLY_RETRY_VERIFICATION.json"


def _require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def verify() -> dict:
    paths = task_files("classification")
    train = pd.read_csv(paths["train"], low_memory=False)
    test = pd.read_csv(paths["test"], low_memory=False, usecols=lambda field: field != "artifact_signal")
    sample = pd.read_csv(paths["sample"], low_memory=False)
    csv = pd.read_csv(retry.CSV_PATH)
    manifest = json.loads(retry.MANIFEST_PATH.read_text(encoding="utf-8"))
    parent_freeze = json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(retry.AMENDMENT_FREEZE.read_text(encoding="utf-8"))
    _require(len(train) == 50_608 and len(test) == len(sample) == len(csv) == 29_421,
             "official train/test/sample/submission row count mismatch")
    _require("artifact_signal" not in test.columns, "Test target was read")
    _require(list(sample.columns) == list(csv.columns) == ["record_id", "artifact_signal"],
             "CSV does not match SampleSubmission schema")
    _require(csv.record_id.astype(str).equals(test.record_id.astype(str)) and
             csv.record_id.astype(str).equals(sample.record_id.astype(str)) and
             not csv.record_id.astype(str).duplicated().any(),
             "CSV IDs are not a one-to-one ordered match to official Test")
    _require(csv.artifact_signal.dtype.kind in "iu" and set(csv.artifact_signal.unique()) <= {0, 1},
             "Classification output is not integer 0/1")
    _require(manifest["csv"]["sha256"] == file_sha256(retry.CSV_PATH) and
             manifest["official_input_sha256"]["train"] == file_sha256(paths["train"]) and
             manifest["official_input_sha256"]["test"] == file_sha256(paths["test"]) and
             manifest["official_input_sha256"]["sample"] == file_sha256(paths["sample"]),
             "CSV or official input hash mismatch")
    _require(manifest["fit_rows"] == 50_608 and manifest["fit_cycle_range"] == [47, 79] and
             manifest["test_target_column_read"] is False and manifest["test_labels_used"] is False and
             manifest["preprocessing_fit_only_train"] is True and manifest["kaggle_called"] is False and
             manifest["submission_ref"] is None, "manifest has incorrect data boundary or submission status")

    identity = {
        "batch_id": retry.TASK, "card_id": retry.RETRY_CARD, "stage": retry.RETRY_STAGE,
        "parent_freeze_signature": parent_freeze["freeze_signature"],
        "technical_retry_amendment": "FULLTRAIN_RETRY_AMENDMENT_01",
        "candidate": "CLASS_ONLY", "cutoff": 79,
    }
    bundle = read_bundle(retry.RETRY_BUNDLE, identity, task="classification")
    events = read_events(retry.LEDGER)
    retry_saved = [e for e in events if e.get("fit_id") == retry.RETRY_ID and e.get("event") == "FIT_SAVED"]
    original_failed = [e for e in events if e.get("fit_id") == "MEASURED_CLASS_ONLY_FULLTRAIN" and
                       e.get("event") == "FIT_FAILED"]
    _require(len(retry_saved) == len(original_failed) == 1 and
             retry_saved[0].get("receipt_sha256") == bundle.receipt["receipt_sha256"] and
             manifest["fit_receipt_sha256"] == bundle.receipt["receipt_sha256"],
             "fit receipt ledger/bundle/manifest lineage mismatch")
    metadata = bundle.receipt["metadata"]["metadata"]
    state = metadata["fit_state"]
    y = train.artifact_signal.to_numpy(dtype=np.int8)
    fit_fp = _fit_fingerprint(train.loc[:, list(FEATURE_NAMES)], tuple(FEATURE_NAMES), y)
    state_without_hash = {key: value for key, value in state.items() if key != "state_sha256"}
    state_sha = hashlib.sha256(json.dumps(state_without_hash, sort_keys=True, separators=(",", ":"),
                                                    allow_nan=False).encode()).hexdigest()
    checks = {
        "fit_rows_cycles_target_fingerprint": (state.get("fit_rows") == 50_608 and
            state.get("fit_cycle_min") == 47 and state.get("fit_cycle_max") == 79 and
            state.get("fit_data_sha256") == fit_fp and isinstance(state.get("target_sha256"), str) and
            bool(state.get("target_sha256"))),
        "fit_state_hash_recomputed": state.get("state_sha256") == state_sha == manifest["fit_state_sha256"],
        "fit_boundary_flags": state.get("test_data_used") is False and
                              state.get("validation_labels_passed_to_fit") is False and
                              metadata.get("test_labels_used") is False,
        "baseline_prediction_persisted": bundle.arrays["baseline_prediction"].shape == (len(test),),
        "preexisting_failed_fit_preserved": len(original_failed) == 1 and
             "baseline_prediction" in original_failed[0].get("error", ""),
        "synthetic_failure_recovery_was_tested": json.loads(retry.SYNTHETIC_REPORT.read_text(encoding="utf-8")).get(
            "post_fit_failure_injection_then_payload_recovery") == "PASS",
    }
    with np.load(fulltrain.CLASS_CACHE, allow_pickle=False) as cache:
        c2b = np.asarray(cache["base"], dtype=np.float64)
        cb = np.asarray(cache["missing"], dtype=np.float64)
        missing = np.asarray(cache["missing_mask"], dtype=bool)
        baseline = np.asarray(cache["prediction"], dtype=np.float64)
    manifest_cache = json.loads(fulltrain.CLASS_MANIFEST.read_text(encoding="utf-8"))
    component_audits = {item["fit_component"]["model"]: item
                        for item in manifest_cache["component_audits"]}
    checks["frozen_cache_hashes"] = (
        measured._array_sha(c2b) == component_audits["C2_B"]["prediction_sha256"] and
        measured._array_sha(cb) == component_audits["CB_MISSING"]["prediction_sha256"] and
        measured._array_sha(baseline) == manifest_cache["prediction_sha256"])
    expected_probability = np.where(missing, cb, 0.75 * c2b + 0.25 * bundle.arrays["raw_prediction"])
    checks["frozen_routing_reconstructed"] = np.allclose(bundle.arrays["prediction"], expected_probability,
                                                       rtol=0, atol=1e-12)
    checks["baseline_is_sparse25_comparator"] = np.array_equal(bundle.arrays["baseline_prediction"], baseline)
    checks["csv_equals_bundle_threshold_050"] = np.array_equal(
        csv.artifact_signal.to_numpy(dtype=np.int8), (bundle.arrays["prediction"] >= 0.50).astype(np.int8))
    temporal = json.loads((measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json").read_text(encoding="utf-8"))
    checks["N02_fulltrain_gate_still_closed"] = (
        temporal.get("decision") == "BENCHMARK_GATES_FAIL" and
        temporal.get("gates", {}).get("primary_gain_vs_sparse25_ge_0_0002") is False and
        not (measured.REPORT_ROOT / "N02_SKETCH_SUBMISSION_MANIFEST.json").exists())
    recovery_events = [e for e in events if e.get("event") == "OUTPUT_PREPARATION_RECOVERED" and
                       e.get("fit_id") == retry.RETRY_ID]
    checks["output_recovery_event_appended_once"] = (
        len(recovery_events) == 1 and
        recovery_events[0].get("csv_sha256") == file_sha256(retry.CSV_PATH) and
        recovery_events[0].get("official_fit_consumed") is False)
    budget = ledger_summary(retry.LEDGER)
    checks["fit_cap_and_retry_accounting"] = (
        budget["authorized_cap"] == 60 and budget["started_and_consumed"] == 40 and
        budget["saved"] == 39 and budget["failed_or_interrupted_started"] == 1 and
        budget["category_reservations"].get("technical_retry") == 1)
    all_checks = all(bool(value) for value in checks.values())
    result = {
        "status": "PASS_INDEPENDENT_ZERO_FIT_VERIFICATION" if all_checks else "FAIL",
        "checks": checks, "fit_ledger": budget,
        "csv_sha256": file_sha256(retry.CSV_PATH),
        "manifest_sha256": file_sha256(retry.MANIFEST_PATH),
        "bundle_receipt_sha256": bundle.receipt["receipt_sha256"],
        "source_commit_at_fit": manifest["source_commit"],
        "postfit_verifier_sha256": file_sha256(Path(__file__)),
        "official_fit_rerun": False, "test_target_read": False, "kaggle_called": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _require(all_checks, json.dumps(result, sort_keys=True))
    measured._write_once(OUTPUT, result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
