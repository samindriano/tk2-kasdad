"""Independent, zero-fit verification of the ClassOnly technical-retry CSV."""
from __future__ import annotations

from datetime import datetime, timezone
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


def verify() -> dict:
    parent, temporal, events = retry._verify_amendment()
    _ = measured._verify_freeze(measured._context())
    manifest = json.loads(retry.MANIFEST_PATH.read_text(encoding="utf-8"))
    csv = pd.read_csv(retry.CSV_PATH)
    from src.utils.paths import task_files

    test_path = task_files("classification")["test"]
    sample_path = task_files("classification")["sample"]
    test = pd.read_csv(test_path, low_memory=False, usecols=lambda column: column != "artifact_signal")
    sample = pd.read_csv(sample_path, low_memory=False)
    checks = {
        "csv_rows": len(csv) == 29_421,
        "csv_columns_match_sample": list(csv.columns) == list(sample.columns) == ["record_id", "artifact_signal"],
        "ids_exact_order": csv.record_id.astype(str).equals(test.record_id.astype(str)) and
                           csv.record_id.astype(str).equals(sample.record_id.astype(str)),
        "ids_unique": not csv.record_id.astype(str).duplicated().any(),
        "labels_integer_binary": csv.artifact_signal.dtype.kind in "iu" and
                                 set(csv.artifact_signal.unique()) <= {0, 1},
        "manifest_csv_hash": manifest["csv"]["sha256"] == file_sha256(retry.CSV_PATH),
        "manifest_source_commit_is_ancestor": measured.n02._is_ancestor(manifest["source_commit"]),
        "manifest_fit_rows": manifest["fit_rows"] == 50_608 and manifest["fit_cycle_range"] == [47, 79],
        "manifest_test_boundary": manifest["test_target_column_read"] is False and
                                  manifest["test_labels_used"] is False and
                                  manifest["preprocessing_fit_only_train"] is True,
        "kaggle_not_called": manifest["kaggle_called"] is False and manifest["submission_ref"] is None,
    }
    identity = {"batch_id": retry.TASK, "card_id": retry.RETRY_CARD, "stage": retry.RETRY_STAGE,
                "parent_freeze_signature": parent["freeze_signature"],
                "technical_retry_amendment": "FULLTRAIN_RETRY_AMENDMENT_01",
                "candidate": "CLASS_ONLY", "cutoff": 79}
    bundle = read_bundle(retry.RETRY_BUNDLE, identity, task="classification")
    with np.load(fulltrain.CLASS_CACHE, allow_pickle=False) as cache:
        c2b = np.asarray(cache["base"], dtype=np.float64)
        cb = np.asarray(cache["missing"], dtype=np.float64)
        missing = np.asarray(cache["missing_mask"], dtype=bool)
        current = np.asarray(cache["prediction"], dtype=np.float64)
    terminal = [e for e in events if e.get("fit_id") == retry.RETRY_ID and e.get("event") == "FIT_SAVED"]
    checks["bundle_receipt_matches_ledger"] = len(terminal) == 1 and terminal[0].get("receipt_sha256") == bundle.receipt["receipt_sha256"]
    checks["required_baseline_prediction"] = bundle.arrays["baseline_prediction"].shape == (29_421,)
    checks["csv_matches_thresholded_bundle"] = np.array_equal(
        csv.artifact_signal.to_numpy(dtype=np.int8),
        (bundle.arrays["prediction"] >= 0.50).astype(np.int8))
    expected_probability = np.where(missing, cb, 0.75 * c2b + 0.25 * bundle.arrays["raw_prediction"])
    checks["frozen_route_reconstruction"] = np.allclose(bundle.arrays["prediction"], expected_probability,
                                                       rtol=0, atol=1e-12)
    checks["baseline_is_frozen_sparse25"] = np.array_equal(bundle.arrays["baseline_prediction"], current)
    checks["n02_candidate_still_closed"] = (
        temporal.get("decision") == "BENCHMARK_GATES_FAIL" and
        not (measured.REPORT_ROOT / "N02_SKETCH_SUBMISSION_MANIFEST.json").exists())
    final_ledger = ledger_summary(retry.LEDGER)
    checks["ledger_fit_counts"] = (
        final_ledger["started_and_consumed"] == 40 and final_ledger["saved"] == 39 and
        final_ledger["failed_or_interrupted_started"] == 1)
    result = {
        "status": "PASS_INDEPENDENT_ZERO_FIT_VERIFICATION" if all(checks.values()) else "FAIL",
        "checks": checks, "ledger": ledger_summary(retry.LEDGER),
        "csv_sha256": file_sha256(retry.CSV_PATH),
        "manifest_sha256": file_sha256(retry.MANIFEST_PATH),
        "official_fit_rerun": False, "test_target_read": False,
        "kaggle_called": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, sort_keys=True))
