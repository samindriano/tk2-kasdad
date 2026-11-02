"""One protocol-authorized technical retry for ClassOnly bundle persistence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_measured_classification as measured  # noqa: E402
from scripts import prepare_c16_r22_fulltrain_submissions as fulltrain  # noqa: E402
from src.models.optimization_v3_batch1 import exclusive_training_slot  # noqa: E402
from src.models.research_next_budget import (  # noqa: E402
    finish_fit, ledger_summary, read_events, release_unused, reserve_card, start_fit,
)
from src.models.research_next_contract import (  # noqa: E402
    canonical_sha256, file_sha256, partition_payload, read_bundle,
    resume_or_publish_bundle, validate_arrays,
)
from src.utils.paths import task_files  # noqa: E402

TASK = "MEASURED_CLASSIFICATION_BENCHMARK_V1"
RETRY_ID = "MEASURED_CLASS_ONLY_FULLTRAIN_RETRY_01"
RETRY_CARD = "MEASURED_CLASS_ONLY_TECHNICAL_RETRY_01"
RETRY_STAGE = "BENCHMARK_FULLTRAIN_TECHNICAL_RETRY"
AMENDMENT = ROOT / "experiments/optimization_v3/measured_benchmark_v1/CLASSIFICATION/FULLTRAIN_RETRY_AMENDMENT_01"
AMENDMENT_FREEZE = AMENDMENT / "freeze.json"
SYNTHETIC_REPORT = ROOT / "reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/CLASS_ONLY_RETRY_SYNTHETIC_PREFLIGHT.json"
RESULT_REPORT = ROOT / "reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/CLASS_ONLY_RETRY_PREPARATION.json"
RETRY_BUNDLE = measured.FIT_ROOT / "FULLTRAIN_RETRIES/CLASS_ONLY_RETRY_01"
CSV_PATH = ROOT / "submissions/optimization_v3/measured_benchmark_v1/class_only.csv"
MANIFEST_PATH = measured.REPORT_ROOT / "CLASS_ONLY_SUBMISSION_MANIFEST.json"
THREADS = measured.THREADS
LOCK = measured.LOCK
LEDGER = measured.LEDGER


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_once(path: Path, payload: Any) -> None:
    measured._write_once(path, payload)


def build_fulltrain_payload(*, record_ids: Any, cycle_code: Any, raw_prediction: Any,
                            baseline_prediction: Any, c2b_probability: Any,
                            cb_probability: Any, stream_missing: Any,
                            fit_state: dict[str, Any], resource_preflight: dict[str, Any],
                            weight_plan_diagnostics: dict[str, Any], fit_rows: int = 50_608,
                            cutoff: int = 79, technical_retry_of: str | None = None
                            ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Single production/synthetic payload builder, including all frozen routes."""
    ids = np.asarray(record_ids, dtype="U")
    cycles = np.asarray(cycle_code, dtype=np.int64)
    raw = np.asarray(raw_prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    c2b = np.asarray(c2b_probability, dtype=np.float64)
    cb = np.asarray(cb_probability, dtype=np.float64)
    missing = np.asarray(stream_missing, dtype=bool)
    n = len(ids)
    _require(n > 0 and all(value.shape == (n,) for value in (cycles, raw, baseline, c2b, cb, missing)),
             "ClassOnly payload inputs are not row-aligned")
    _require(all(np.isfinite(value).all() for value in (raw, baseline, c2b, cb)) and
             all(np.all((value >= 0.0) & (value <= 1.0)) for value in (raw, baseline, c2b, cb)),
             "ClassOnly payload probabilities are invalid")
    probability = np.where(missing, cb, 0.75 * c2b + 0.25 * raw).astype(np.float64)
    labels = (probability >= 0.50).astype(np.int8)
    metadata = {
        "candidate": "CLASS_ONLY", "fit_rows": int(fit_rows), "fit_cycle_min": 47,
        "fit_cycle_max": int(cutoff), "cutoff": int(cutoff), "fit_state": fit_state,
        "resource_preflight": resource_preflight,
        "weight_plan_diagnostics": weight_plan_diagnostics,
        "preprocessing_fit_only_official_train": True,
        "test_data_used_only_for_inference": True, "test_target_column_read": False,
        "test_labels_used": False, "threshold": 0.50, "kaggle_called": False,
        "technical_retry_of": technical_retry_of,
        "technical_change": "persist required baseline_prediction array" if technical_retry_of else None,
    }
    arrays = {
        "score_rows": np.arange(n, dtype=np.int64), "score_record_ids": ids,
        "cycle_code": cycles, "prediction": probability,
        "baseline_prediction": baseline, "raw_prediction": raw,
        "label": labels, "stream_missing": missing,
        "metadata": metadata,
    }
    return arrays, probability, labels


class InjectedPostFitFailure(RuntimeError):
    """Synthetic fault inserted after estimator fit/payload build but before publish."""


def _publish_payload(destination: Path, payload: dict[str, Any], identity: dict[str, Any],
                    *, inject_postfit_failure: bool = False):
    arrays, _ = partition_payload(payload)
    validate_arrays(arrays, task="classification")
    if inject_postfit_failure:
        raise InjectedPostFitFailure("synthetic failure after fit and payload build, before artifact publish")
    return resume_or_publish_bundle(destination, payload, identity, task="classification")


def synthetic_payload() -> tuple[dict[str, Any], int]:
    """Fit one synthetic estimator and pass its output through the production payload builder."""
    x = np.arange(8, dtype=np.float64).reshape(-1, 1)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    estimator = DummyClassifier(strategy="prior").fit(x, y)
    raw = estimator.predict_proba(x)[:, 1].astype(np.float64)
    n = len(raw)
    missing = np.asarray([False, True, False, True, False, True, False, True])
    c2b = np.linspace(0.2, 0.9, n, dtype=np.float64)
    cb = np.linspace(0.1, 0.8, n, dtype=np.float64)
    baseline = np.linspace(0.05, 0.95, n, dtype=np.float64)
    payload, _, _ = build_fulltrain_payload(
        record_ids=[f"SYN-{i}" for i in range(n)], cycle_code=np.full(n, 79),
        raw_prediction=raw, baseline_prediction=baseline, c2b_probability=c2b,
        cb_probability=cb, stream_missing=missing,
        fit_state={"synthetic_fit_rows": n, "synthetic_only": True},
        resource_preflight={"status": "SYNTHETIC"},
        weight_plan_diagnostics={"synthetic_only": True}, fit_rows=n,
    )
    payload["metadata"]["synthetic_only"] = True
    return payload, 1


def run_synthetic_preflight() -> dict[str, Any]:
    payload, fit_calls = synthetic_payload()
    identity = {"task_id": TASK, "amendment": "FULLTRAIN_RETRY_AMENDMENT_01",
                "candidate": "CLASS_ONLY", "synthetic": True}
    with tempfile.TemporaryDirectory(prefix="kasdad_classonly_retry_") as temp:
        destination = Path(temp) / "bundle"
        try:
            _publish_payload(destination, payload, identity, inject_postfit_failure=True)
        except InjectedPostFitFailure:
            pass
        else:
            raise RuntimeError("synthetic post-fit failure injection did not fire")
        first = _publish_payload(destination, payload, identity)
        readback = read_bundle(destination, identity, task="classification")
        second = resume_or_publish_bundle(destination, payload, identity, task="classification")
        _require(first.receipt["receipt_sha256"] == readback.receipt["receipt_sha256"] ==
                 second.receipt["receipt_sha256"], "synthetic bundle persistence/resume receipt differs")
        _require(np.array_equal(readback.arrays["baseline_prediction"], payload["baseline_prediction"]),
                 "synthetic required baseline array changed")
        _require(np.array_equal(readback.arrays["prediction"], payload["prediction"]),
                 "synthetic prediction changed")
    report = {
        "status": "PASS",
        "scope": "synthetic estimator only; no official Train/Test or Kaggle",
        "missing_contract_field_repaired": "baseline_prediction",
        "synthetic_rows": len(payload["score_rows"]),
        "synthetic_fit_calls": fit_calls,
        "post_fit_failure_injection_then_payload_recovery": "PASS",
        "bundle_persist_readback_resume": "PASS",
        "receipt_sha256": first.receipt["receipt_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(SYNTHETIC_REPORT, report)
    return report


def _parent_state(*, load_train: bool = True) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    context = measured._context()
    parent = measured._verify_freeze(context)
    _require(SYNTHETIC_REPORT.is_file(), "technical retry synthetic persistence check is missing")
    synthetic = json.loads(SYNTHETIC_REPORT.read_text(encoding="utf-8"))
    _require(synthetic.get("status") == "PASS" and
             synthetic.get("bundle_persist_readback_resume") == "PASS",
             "technical retry synthetic persistence check did not pass")
    temporal = json.loads((measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json").read_text(encoding="utf-8"))
    _require(temporal.get("freeze_signature") == parent["freeze_signature"] and
             temporal.get("decision") == "BENCHMARK_GATES_FAIL" and
             temporal.get("gates", {}).get("primary_gain_vs_sparse25_ge_0_0002") is False,
             "N02 gate decision changed; ClassOnly retry is no longer authorized under this amendment")
    events = read_events(LEDGER)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and
              e.get("fit_id") == "MEASURED_CLASS_ONLY_FULLTRAIN"]
    terminals = [e for e in events if e.get("fit_id") == "MEASURED_CLASS_ONLY_FULLTRAIN" and
                 e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    _require(len(starts) == 1 and len(terminals) == 1 and terminals[0].get("event") == "FIT_FAILED" and
             "Missing required arrays: ['baseline_prediction']" in str(terminals[0].get("error")),
             "expected original ClassOnly persistence failure is absent or changed")
    _require(not RETRY_BUNDLE.exists() and not CSV_PATH.exists() and not MANIFEST_PATH.exists(),
             "ClassOnly retry output already exists; refusing another fit or overwrite")
    _require(not (measured.FIT_ROOT / "FULLTRAIN/CLASS_ONLY").exists(),
             "original ClassOnly full-Train model bundle exists; recovery audit must inspect it before retry")
    return parent, temporal, events


def _amendment_identity() -> dict[str, Any]:
    parent, _, events = _parent_state()
    context = measured._context()
    _, test, _, official_hashes = _read_official_inputs()
    source_files = (
        "scripts/optimization_v3_measured_classification_retry.py",
        "scripts/optimization_v3_measured_classification_retry_verify.py",
        "tests/test_optimization_v3_measured_classification_retry.py",
    )
    return {
        "schema_version": 1,
        "amendment_id": "FULLTRAIN_RETRY_AMENDMENT_01",
        "status": "FROZEN_TECHNICAL_RETRY_ONLY",
        "prepared_at_head": measured._head(),
        "parent_freeze_signature": parent["freeze_signature"],
        "parent_freeze_sha256": file_sha256(measured.FREEZE_PATH),
        "protocol_sha256": file_sha256(measured.PROTOCOL),
        "temporal_result_sha256": file_sha256(measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json"),
        "synthetic_preflight_sha256": file_sha256(SYNTHETIC_REPORT),
        "official_train_sha256": file_sha256(context["train_path"]),
        "official_input_sha256": official_hashes,
        "official_test_ordered_ids_sha256": hashlib.sha256(
            ("\n".join(test.record_id.astype(str)) + "\n").encode("utf-8")).hexdigest(),
        "retry_source_sha256": {name: file_sha256(ROOT / name) for name in source_files},
        "original_failed_fit": {"fit_id": "MEASURED_CLASS_ONLY_FULLTRAIN",
                                 "event": next(e for e in events if e.get("fit_id") == "MEASURED_CLASS_ONLY_FULLTRAIN"
                                               and e.get("event") == "FIT_FAILED")},
        "recovery_audit": {
            "original_runner_process_completed_with_exception": True,
            "original_fulltrain_model_bundle_exists": False,
            "original_class_only_csv_exists": False,
            "original_class_only_manifest_exists": False,
            "original_class_only_retry_bundle_exists": False,
            "expert_raw_prediction_artifact_exists": False,
            "attested_frozen_c2b_cb_sparse25_controls_remain_available": True,
            "recovery_conclusion": "the reused baselines are recoverable, but the newly fitted ClassOnly estimator and its raw Test probabilities were never persisted; one technical retry is required",
        },
        "technical_retry": {
            "fit_id": RETRY_ID, "card_id": RETRY_CARD, "stage": RETRY_STAGE,
            "category": "technical_retry", "maximum_fits": 1,
            "recipe_unchanged": True,
            "only_change": "persist the existing comparison baseline in the required baseline_prediction bundle field",
            "estimator": "N01 exact class_only sparse recipe; full official Train cycles 47-79",
            "blend": "present .75 C2_B + .25 class-only sparse expert; missing frozen CB_MISSING",
            "threshold": 0.50,
        },
        "budget_before_retry": ledger_summary(LEDGER),
        "kaggle_called": False,
    }


def freeze_amendment() -> dict[str, Any]:
    identity = _amendment_identity()
    frozen = {"identity": identity, "freeze_signature": canonical_sha256(identity)}
    _write_once(AMENDMENT_FREEZE, frozen)
    return frozen


def _verify_amendment() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    parent, temporal, events = _parent_state()
    frozen = json.loads(AMENDMENT_FREEZE.read_text(encoding="utf-8"))
    identity = frozen.get("identity", {})
    _require(frozen.get("freeze_signature") == canonical_sha256(identity),
             "technical retry amendment signature is invalid")
    _require(identity.get("parent_freeze_signature") == parent.get("freeze_signature"),
             "technical retry parent freeze changed")
    _require(identity.get("prepared_at_head") and measured.n02._is_ancestor(identity["prepared_at_head"]),
             "technical retry preparation commit is not an ancestor of current HEAD")
    for name, expected in identity.get("retry_source_sha256", {}).items():
        _require(file_sha256(ROOT / name) == expected, f"technical retry source changed after freeze: {name}")
    _require(file_sha256(measured.FREEZE_PATH) == identity.get("parent_freeze_sha256") and
             file_sha256(measured.PROTOCOL) == identity.get("protocol_sha256") and
             file_sha256(measured.REPORT_ROOT / "N02_CONTINUATION_RESULTS.json") ==
             identity.get("temporal_result_sha256") and
             file_sha256(SYNTHETIC_REPORT) == identity.get("synthetic_preflight_sha256"),
             "technical retry frozen inputs changed")
    _, test, _, current_hashes = _read_official_inputs()
    _require(current_hashes == identity.get("official_input_sha256") and
             hashlib.sha256(("\n".join(test.record_id.astype(str)) + "\n").encode("utf-8")).hexdigest() ==
             identity.get("official_test_ordered_ids_sha256"),
             "official inference inputs or ordered Test IDs changed after technical retry freeze")
    return parent, temporal, events


def _read_official_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    files = task_files("classification")
    train_path, test_path, sample_path = files["train"], files["test"], files["sample"]
    context = measured._context()
    _require(train_path == context["train_path"], "official Train path differs from frozen measured source")
    train = pd.read_csv(train_path, low_memory=False)
    # Exclude the target column at CSV parse time; Test is used only for inference and ID QA.
    test = pd.read_csv(test_path, low_memory=False,
                       usecols=lambda column: column != "artifact_signal")
    sample = pd.read_csv(sample_path, low_memory=False)
    _require(len(train) == 50_608 and len(test) == 29_421 and len(sample) == 29_421,
             "official Train/Test/Sample row counts changed")
    _require("artifact_signal" not in test.columns and list(sample.columns) == ["record_id", "artifact_signal"],
             "Test target was read or official Sample schema changed")
    _require(test.record_id.astype(str).equals(sample.record_id.astype(str)),
             "official Test IDs differ from SampleSubmission order")
    hashes = {"train": file_sha256(train_path), "test": file_sha256(test_path),
              "sample": file_sha256(sample_path)}
    _require(hashes["train"] == json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))
             ["identity"]["input_sha256"]["data/raw/classification/Train.csv"],
             "official Train bytes differ from the measured freeze")
    return train, test, sample, hashes


def _reuse_current(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                       np.ndarray, dict[str, Any]]:
    c2b, cb, missing, reuse_audit = fulltrain._verify_classification_reuse(train, test)
    with np.load(fulltrain.CLASS_CACHE, allow_pickle=False) as cache:
        current = np.asarray(cache["prediction"], dtype=np.float64)
    cache_manifest = json.loads(fulltrain.CLASS_MANIFEST.read_text(encoding="utf-8"))
    _require(measured._array_sha(current) == cache_manifest.get("prediction_sha256") and
             current.shape == (len(test),) and np.isfinite(current).all() and
             np.all((current >= 0.0) & (current <= 1.0)), "frozen Sparse25 baseline cache audit failed")
    _require(missing.shape == cb.shape == c2b.shape == current.shape == (len(test),),
             "reused full-Train inference arrays are misaligned")
    return c2b, cb, missing, current, reuse_audit


def _run_one_fit(train: pd.DataFrame, test: pd.DataFrame, c2b: np.ndarray, cb: np.ndarray,
                 missing: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    events = read_events(LEDGER)
    _require(not any(e.get("event") == "FIT_STARTED" and e.get("fit_id") == RETRY_ID for e in events),
             "the one authorized retry was already started; do not refit")
    retry_starts = [e for e in events if e.get("event") == "FIT_STARTED" and
                    str(e.get("card_id", "")).startswith("MEASURED_CLASS_ONLY_TECHNICAL_RETRY")]
    _require(len(retry_starts) == 0, "a ClassOnly technical retry already consumed its one-retry allowance")
    y = train.artifact_signal.to_numpy(dtype=np.int8)
    _require(len(y) == 50_608 and set(np.unique(y)) == {0, 1}, "official Train target/population failed audit")
    features = train.loc[:, list(measured.FEATURES)]
    score_features = test.loc[:, list(measured.FEATURES)]
    plan = measured.class_by_volume_weight_plan(features, y)
    resource = {"available_physical_bytes": int(psutil.virtual_memory().available),
                "estimated_peak_bytes": 2 * 1024**3, "job_cap_bytes": measured.JOB_CAP_BYTES,
                "reserve_bytes": measured.RESERVE_BYTES, "threads": THREADS}
    _require(resource["available_physical_bytes"] >= measured.MIN_FREE_BYTES,
             f"ClassOnly technical retry resource preflight failed: {resource}")
    reserve_card(LEDGER, card_id=RETRY_CARD, stage=RETRY_STAGE, fits=1, category="technical_retry")
    started = False
    try:
        with exclusive_training_slot(LOCK):
            fresh = int(psutil.virtual_memory().available)
            _require(fresh >= measured.MIN_FREE_BYTES + measured.RESERVE_BYTES,
                     f"in-lock ClassOnly technical retry memory below guard: {fresh}")
            resource["in_lock_available_physical_bytes"] = fresh
            resource["status"] = "PASS"
            start_fit(LEDGER, fit_id=RETRY_ID, card_id=RETRY_CARD, stage=RETRY_STAGE,
                      cutoff=79, component="CLASS_ONLY_TECHNICAL_RETRY")
            started = True
            with threadpool_limits(limits=THREADS):
                model = measured.FullTrainClassOnly().fit(features, y, sample_weight=plan["class_only_weights"])
                raw = model.predict_probability(score_features)
            probability = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
            _require(probability.shape == (len(test),) and np.isfinite(probability).all() and
                     np.all((probability >= 0) & (probability <= 1)), "ClassOnly retry probabilities invalid")
            labels = (probability >= 0.50).astype(np.int8)
            arrays, probability, labels = build_fulltrain_payload(
                record_ids=test.record_id.astype(str).to_numpy(dtype="U"),
                cycle_code=pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64),
                raw_prediction=raw, baseline_prediction=current, c2b_probability=c2b,
                cb_probability=cb, stream_missing=missing, fit_state=model.audit_state(),
                resource_preflight=resource, weight_plan_diagnostics=plan["diagnostics"],
                fit_rows=len(train), cutoff=79, technical_retry_of="MEASURED_CLASS_ONLY_FULLTRAIN",
            )
            parent = json.loads(measured.FREEZE_PATH.read_text(encoding="utf-8"))
            identity = {"batch_id": TASK, "card_id": RETRY_CARD, "stage": RETRY_STAGE,
                        "parent_freeze_signature": parent["freeze_signature"],
                        "technical_retry_amendment": "FULLTRAIN_RETRY_AMENDMENT_01",
                        "candidate": "CLASS_ONLY", "cutoff": 79}
            bundle = _publish_payload(RETRY_BUNDLE, arrays, identity)
            finish_fit(LEDGER, fit_id=RETRY_ID, status="FIT_SAVED",
                       receipt_sha256=bundle.receipt["receipt_sha256"])
            return {"candidate": "CLASS_ONLY", "fit_id": RETRY_ID,
                    "bundle": measured._relative(RETRY_BUNDLE),
                    "receipt_sha256": bundle.receipt["receipt_sha256"],
                    "fit_state": model.audit_state(), "probability": probability,
                    "labels": labels, "raw": raw, "resource_preflight": resource,
                    "plan_diagnostics": plan["diagnostics"], "train_rows": len(train),
                    "baseline_prediction": current}
    except Exception as exc:
        if started:
            now = read_events(LEDGER)
            terminal = [e for e in now if e.get("fit_id") == RETRY_ID and
                        e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
            if not terminal:
                finish_fit(LEDGER, fit_id=RETRY_ID, status="FIT_FAILED",
                           error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        release_unused(LEDGER, card_id=RETRY_CARD, stage=RETRY_STAGE)


def _prepare_csv_and_manifest(fit: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame,
                              sample: pd.DataFrame, hashes: dict[str, str],
                              reuse_audit: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    frame = measured._submission_frame(test, sample, fit["labels"], "artifact_signal")
    _require(not CSV_PATH.exists() and not MANIFEST_PATH.exists(),
             "ClassOnly output already exists; refusing overwrite or duplicate")
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    measured._write_once(CSV_PATH, payload)
    readback = pd.read_csv(CSV_PATH)
    ids = test.record_id.astype(str).reset_index(drop=True)
    _require(len(readback) == 29_421 and list(readback.columns) == ["record_id", "artifact_signal"] and
             readback.record_id.astype(str).equals(ids) and
             not readback.record_id.astype(str).duplicated().any() and
             np.array_equal(readback.artifact_signal.to_numpy(dtype=np.int64), fit["labels"]) and
             set(np.unique(readback.artifact_signal.to_numpy(dtype=np.int64))) <= {0, 1},
             "ClassOnly CSV round-trip QA failed")
    input_sha = {**hashes,
                 "original_measured_freeze": file_sha256(measured.FREEZE_PATH),
                 "measured_protocol": file_sha256(measured.PROTOCOL),
                 "technical_retry_amendment_freeze": file_sha256(AMENDMENT_FREEZE)}
    csv_sha = file_sha256(CSV_PATH)
    manifest = {
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED", "task": "classification",
        "candidate": "CLASS_ONLY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": measured._head(), "parent_freeze_signature": amendment["identity"]["parent_freeze_signature"],
        "technical_retry_amendment_signature": amendment["freeze_signature"],
        "technical_retry_of": "MEASURED_CLASS_ONLY_FULLTRAIN",
        "recipe": "N01 exact class_only sparse recipe; all official Train cycles 47-79; present .75 C2_B + .25 weighted sparse expert; missing frozen CB_MISSING; threshold 0.50",
        "official_input_sha256": input_sha, "fit_rows": 50_608, "fit_cycle_range": [47, 79],
        "test_rows": 29_421, "fit_state_sha256": fit["fit_state"]["state_sha256"],
        "weight_plan": fit["plan_diagnostics"],
        "fit_bundle": fit["bundle"], "fit_receipt_sha256": fit["receipt_sha256"],
        "fit_ledger": ledger_summary(LEDGER), "reuse_audit": reuse_audit,
        "test_target_column_read": False, "test_labels_used": False,
        "preprocessing_fit_only_train": True,
        "csv": {"path_gitignored": CSV_PATH.relative_to(ROOT).as_posix(),
                "sha256": csv_sha, "rows": len(frame), "columns": list(frame.columns),
                "ordered_ids_sha256": hashlib.sha256(("\n".join(frame.record_id.astype(str)) + "\n").encode()).hexdigest(),
                "labels": {str(int(k)): int(v) for k, v in zip(*np.unique(fit["labels"], return_counts=True))}},
        "kaggle_called": False, "submission_ref": None,
    }
    _write_once(MANIFEST_PATH, manifest)
    report = {
        "status": "CLASS_ONLY_CSV_PREPARED_VALIDATED_NOT_SUBMITTED",
        "candidate": manifest["candidate"], "csv": manifest["csv"],
        "fit_bundle": manifest["fit_bundle"], "fit_receipt_sha256": manifest["fit_receipt_sha256"],
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "technical_retry_of": "MEASURED_CLASS_ONLY_FULLTRAIN",
        "N02_SKETCH": "NO_CSV_BENCHMARK_GATES_FAILED",
        "test_labels_used": False, "kaggle_called": False,
        "fit_ledger": ledger_summary(LEDGER),
    }
    _write_once(RESULT_REPORT, report)
    return report


def run_fulltrain_retry() -> dict[str, Any]:
    parent, temporal, events = _verify_amendment()
    train, test, sample, hashes = _read_official_inputs()
    c2b, cb, missing, current, reuse = _reuse_current(train, test)
    if RETRY_BUNDLE.exists():
        starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == RETRY_ID]
        terminal = [e for e in events if e.get("fit_id") == RETRY_ID and e.get("event") == "FIT_SAVED"]
        _require(len(starts) == len(terminal) == 1, "existing retry bundle lacks a saved fit receipt")
        identity = {"batch_id": TASK, "card_id": RETRY_CARD, "stage": RETRY_STAGE,
                    "parent_freeze_signature": parent["freeze_signature"],
                    "technical_retry_amendment": "FULLTRAIN_RETRY_AMENDMENT_01",
                    "candidate": "CLASS_ONLY", "cutoff": 79}
        bundle = read_bundle(RETRY_BUNDLE, identity, task="classification")
        _require(bundle.receipt["receipt_sha256"] == terminal[0].get("receipt_sha256"),
                 "existing retry bundle receipt differs from fit ledger")
        metadata = bundle.receipt.get("metadata", {}).get("metadata", {})
        fit = {"candidate": "CLASS_ONLY", "fit_id": RETRY_ID,
               "bundle": measured._relative(RETRY_BUNDLE), "receipt_sha256": bundle.receipt["receipt_sha256"],
               "fit_state": metadata["fit_state"], "probability": bundle.arrays["prediction"],
               "labels": bundle.arrays["label"], "raw": bundle.arrays["raw_prediction"],
               "resource_preflight": metadata["resource_preflight"],
               "plan_diagnostics": metadata["weight_plan_diagnostics"],
               "train_rows": 50_608, "baseline_prediction": bundle.arrays["baseline_prediction"]}
    else:
        fit = _run_one_fit(train, test, c2b, cb, missing, current)
    report = _prepare_csv_and_manifest(fit, train, test, sample, hashes, reuse, parent)
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("synthetic", "freeze", "fulltrain"))
    action = parser.parse_args().action
    if action == "synthetic":
        print(json.dumps(run_synthetic_preflight(), sort_keys=True))
    elif action == "freeze":
        print(json.dumps(freeze_amendment(), sort_keys=True))
    else:
        run_fulltrain_retry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
