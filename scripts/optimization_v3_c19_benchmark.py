"""Execute the one frozen C19 full-Train benchmark and prepare its CSV.

The model recipe is inherited without edits from
``src.models.optimization_v3_batch1_c19``.  This runner uses an attested
full-Train sparse25 Test-side cache and never reads Test labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submit_regression_research import ordered_ids_sha256  # noqa: E402
from src.models.classification_c19_fulltrain import C19FullTrainClassifier  # noqa: E402
from src.models.classification_round2_models import stream_missing_mask  # noqa: E402
from src.models.optimization_v3_batch1_c19 import (  # noqa: E402
    BLEND_WEIGHT,
    FEATURE_NAMES,
    THRESHOLD,
    XGB_PARAMETERS,
    XGBoostSVD32Classifier,
)
from src.models.research_next_budget import finish_fit, read_events, start_fit  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from src.models.round1_models import CATEGORICAL_FEATURES, EXPANDED_FEATURES  # noqa: E402


PROTOCOL_PATH = ROOT / "experiments/optimization_v3/c19_a02_v1/C19/protocol.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/c19_a02_v1/C19/freeze.json"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_c19_benchmark_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/c19_benchmark_v1"
TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
TEST_PATH = ROOT / "data/raw/classification/Test.csv"
SAMPLE_PATH = ROOT / "data/raw/classification/SampleSubmission.csv"
BASE_CACHE = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"
BASE_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
BASE_CSV = ROOT / "submissions/dual_campaign/optimization_v3/v3_cls_sparse25.csv"
BASE_CACHE_SHA256 = "31ef6945ae74a14b1e4ff46220667ac3a08de3401120257e7932745c927d6103"
BASE_MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
BASE_CSV_SHA256 = "bc926edcb3258ec76ed8334db64fcb6ae1382e849a8128b9d88d1f9d342595b2"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_TEST_ROWS = 29_421
EXPECTED_TRAIN_ROWS = 50_608
COMPETITION = "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027"
SUBMISSION_MESSAGE = "Samuel C19 XGB SVD32 Frozen Benchmark"
CARD_ID = "CLASSIFICATION_C19_FULLTRAIN_BENCHMARK"
STAGE = "C19_FULLTRAIN_BENCHMARK"
FIT_ID = "C19_FULLTRAIN_XGB_SVD32"
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
MIN_FREE_RAM_BYTES = 4 * 1024**3


def _sha(path: Path) -> str:
    return file_sha256(path)


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Immutable output differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_frozen() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _read_json(PROTOCOL_PATH)
    freeze = _read_json(FREEZE_PATH)
    if canonical_sha256(protocol) != freeze.get("protocol_sha256"):
        raise RuntimeError("C19 protocol hash differs from the frozen identity")
    identity = freeze.get("identity")
    if not isinstance(identity, dict) or canonical_sha256(identity) != freeze.get("freeze_signature"):
        raise RuntimeError("C19 freeze signature is invalid")
    for path_text, expected in freeze["source_sha256"].items():
        path = ROOT / path_text
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"C19 frozen source hash mismatch: {path_text}")
    for path_text, expected in freeze["input_sha256"].items():
        path = ROOT / path_text
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"C19 frozen input hash mismatch: {path_text}")
    return protocol, freeze


def _load_train_test_sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol, _freeze = _load_frozen()
    feature_names = list(protocol["feature_allowlist"])
    train = pd.read_csv(TRAIN_PATH, usecols=["record_id", "artifact_signal", *feature_names],
                         dtype={"record_id": str})
    # The Test target column is not included in usecols and is never loaded.
    test = pd.read_csv(TEST_PATH, usecols=["record_id", *feature_names], dtype={"record_id": str})
    sample = pd.read_csv(SAMPLE_PATH, dtype={"record_id": str})
    return train, test, sample


def _load_baseline_cache(test: pd.DataFrame, sample: pd.DataFrame) -> dict[str, np.ndarray]:
    if _sha(BASE_CACHE) != BASE_CACHE_SHA256:
        raise RuntimeError("Full-Train sparse25 cache SHA-256 changed")
    if _sha(BASE_MANIFEST) != BASE_MANIFEST_SHA256:
        raise RuntimeError("Full-Train sparse25 manifest SHA-256 changed")
    if _sha(BASE_CSV) != BASE_CSV_SHA256:
        raise RuntimeError("Full-Train sparse25 CSV SHA-256 changed")
    manifest = _read_json(BASE_MANIFEST)
    if (manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("recipe") != {
                "kind": "availability_present_blend",
                "present": [
                    {"model": "C2_B", "weight": 0.75},
                    {"model": "SPARSE_BINARY", "weight": 0.25},
                ],
                "missing": "CB_MISSING",
                "threshold": 0.5,
            }
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("test_labels_used") is not False):
        raise RuntimeError("Existing sparse25 manifest does not attest the frozen baseline recipe")
    test_ids = test["record_id"].astype(str).tolist()
    sample_ids = sample["record_id"].astype(str).tolist()
    if (len(test_ids) != EXPECTED_TEST_ROWS or len(set(test_ids)) != len(test_ids)
            or test_ids != sample_ids
            or ordered_ids_sha256(test_ids) != manifest.get("ordered_test_ids_sha256")):
        raise RuntimeError("Test IDs/order differ from the frozen sparse25 cache manifest")
    baseline_frame = pd.read_csv(BASE_CSV, dtype={"record_id": str})
    if (baseline_frame.columns.tolist() != ["record_id", "artifact_signal"]
            or baseline_frame["record_id"].tolist() != test_ids):
        raise RuntimeError("The preserved baseline CSV does not bind cache order to Test IDs")

    with np.load(BASE_CACHE, allow_pickle=False) as archive:
        required = {"prediction", "probability", "base", "missing", "sparse", "missing_mask"}
        if set(archive.files) != required:
            raise RuntimeError("Full-Train sparse25 cache schema changed")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    if any(value.shape != (EXPECTED_TEST_ROWS,) for value in arrays.values()):
        raise RuntimeError("Full-Train sparse25 cache has an incorrect row count")
    for name in ("probability", "base", "missing", "sparse"):
        value = arrays[name]
        if value.dtype.kind != "f" or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise RuntimeError(f"Invalid cached baseline probability array: {name}")
    mask = arrays["missing_mask"]
    if mask.dtype != np.bool_ or not np.array_equal(mask, stream_missing_mask(test)):
        raise RuntimeError("Cached sparse25 missing-stream mask differs from official Test features")
    reconstructed = np.where(mask, arrays["missing"], 0.75 * arrays["base"] + 0.25 * arrays["sparse"])
    if not np.array_equal(reconstructed, arrays["probability"]):
        raise RuntimeError("Cached sparse25 probabilities do not reconstruct bit-exactly")
    labels = (arrays["probability"] >= THRESHOLD).astype(np.int8)
    if (arrays["prediction"].dtype != np.int8
            or not np.array_equal(labels, arrays["prediction"])
            or not np.array_equal(labels, baseline_frame["artifact_signal"].to_numpy(dtype=np.int8))):
        raise RuntimeError("Cached sparse25 hard labels do not match its probabilities and ordered CSV")
    return arrays


class _PeakRSS:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak = 0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRSS":
        process = psutil.Process()
        self.peak = process.memory_info().rss

        def monitor() -> None:
            while not self._stop.wait(0.2):
                try:
                    self.peak = max(self.peak, process.memory_info().rss)
                except psutil.Error:
                    return

        self._thread = threading.Thread(target=monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak = max(self.peak, psutil.Process().memory_info().rss)
        except psutil.Error:
            pass


def _preflight() -> dict[str, Any]:
    protocol, freeze = _load_frozen()
    train, test, sample = _load_train_test_sample()
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise RuntimeError("C19 official train/test row counts differ from frozen expectations")
    if train.columns.has_duplicates or test.columns.has_duplicates or sample.columns.has_duplicates:
        raise RuntimeError("C19 input schema contains duplicate columns")
    if set(protocol["feature_allowlist"]) != set(FEATURE_NAMES):
        raise RuntimeError("C19 frozen feature allowlist differs from the historical estimator source")
    if {"artifact_signal", "threat_rank"}.intersection(test.columns):
        raise RuntimeError("C19 Test target was loaded")
    if sample.columns.tolist() != ["record_id", "artifact_signal"]:
        raise RuntimeError("C19 SampleSubmission schema differs from the target task")
    if train["record_id"].isna().any() or train["record_id"].duplicated().any():
        raise RuntimeError("C19 Train IDs are null or duplicated")
    cycles = pd.to_numeric(train["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64))
            or not np.array_equal(np.unique(train["artifact_signal"].to_numpy()), np.array([0, 1]))):
        raise RuntimeError("C19 full-Train cycle or target coverage differs from the frozen population")
    baseline = _load_baseline_cache(test, sample)
    test_ids = test["record_id"].astype(str).tolist()
    if test["record_id"].isna().any() or test["record_id"].duplicated().any():
        raise RuntimeError("C19 official Test IDs are null or duplicated")

    # A conservative, explicit upper bound: 254 OHE categories per categorical
    # field, SVD32 plus missing indicator, and 512 MiB for XGBoost/runtime state.
    structured_width = (len(EXPANDED_FEATURES) - len(CATEGORICAL_FEATURES)
                        + len(CATEGORICAL_FEATURES) * 254)
    design_width_upper = structured_width + 32 + 1
    dense_design_bytes = len(train) * design_width_upper * 8
    analyzer = TfidfVectorizer(**dict(protocol["tfidf_parameters"])).build_analyzer()
    text_values = train["signal_stream"].fillna("").astype(str).tolist()
    nnz_upper = sum(len(analyzer(text)) for text in text_values)
    tfidf_sparse_bytes_upper = nnz_upper * (8 + 4) + (len(train) + 1) * 4
    estimated_peak = 2 * dense_design_bytes + tfidf_sparse_bytes_upper + 512 * 1024**2
    virtual = psutil.virtual_memory()
    disk = shutil.disk_usage(RUN_ROOT.parent if RUN_ROOT.parent.exists() else ROOT)
    if virtual.available < MIN_FREE_RAM_BYTES:
        raise RuntimeError(f"C19 preflight free RAM {virtual.available} is below the 4 GiB minimum")
    if estimated_peak > JOB_CAP_BYTES:
        raise RuntimeError(f"C19 conservative peak estimate {estimated_peak} exceeds the 6 GiB job cap")
    if disk.free < 2 * JOB_CAP_BYTES:
        raise RuntimeError("C19 preflight has less than 12 GiB free for durable state and recovery")
    report = {
        "status": "C19_PREFLIGHT_PASS",
        "campaign": "C19_XGB_SVD32_FULLTRAIN_BENCHMARK",
        "freeze_signature": freeze["freeze_signature"],
        "protocol_sha256": freeze["protocol_sha256"],
        "official_train_rows": len(train),
        "train_cycle_min": int(cycles.min()),
        "train_cycle_max": int(cycles.max()),
        "test_rows": len(test),
        "test_target_column_loaded": False,
        "sample_rows": len(sample),
        "test_ids_sha256": ordered_ids_sha256(test_ids),
        "sample_ids_sha256": ordered_ids_sha256(sample["record_id"].astype(str).tolist()),
        "baseline_cache_sha256": _sha(BASE_CACHE),
        "baseline_cache_route_reconstructed_bit_exact": True,
        "baseline_missing_mask_count": int(baseline["missing_mask"].sum()),
        "baseline_present_mask_count": int((~baseline["missing_mask"]).sum()),
        "feature_allowlist": list(protocol["feature_allowlist"]),
        "resource_preflight": {
            "concurrent_training_jobs": 1,
            "threads": THREADS,
            "job_memory_cap_bytes": JOB_CAP_BYTES,
            "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES,
            "available_ram_bytes": int(virtual.available),
            "estimated_structured_plus_svd_design_width_upper": int(design_width_upper),
            "estimated_dense_design_bytes": int(dense_design_bytes),
            "tfidf_analyzer_nnz_upper_bound": int(nnz_upper),
            "estimated_tfidf_csr_bytes_upper": int(tfidf_sparse_bytes_upper),
            "additional_xgb_runtime_reserve_bytes": 512 * 1024**2,
            "estimated_peak_bytes": int(estimated_peak),
            "free_disk_bytes": int(disk.free),
            "status": "PASS",
        },
        "recipe_source_class": "src.models.optimization_v3_batch1_c19.XGBoostSVD32Classifier",
        "fulltrain_wrapper": "src.models.classification_c19_fulltrain.C19FullTrainClassifier",
        "parameters_match_historical_source": XGB_PARAMETERS == protocol["xgb_parameters"],
        "protocol_boundary": {
            "fit_population": "official Train only, cycles 47-79",
            "test_features_only_for_inference": True,
            "test_labels_used": False,
            "other_task_labels_used": False,
            "preprocessing_fit_only_on_train": True,
        },
    }
    if report["parameters_match_historical_source"] is not True:
        raise RuntimeError("C19 XGBoost parameters differ from the frozen historical recipe")
    preflight_path = REPORT_ROOT / "official_preflight.json"
    if preflight_path.exists():
        previous = _read_json(preflight_path)
        stable_fields = (
            "status", "campaign", "freeze_signature", "protocol_sha256", "official_train_rows",
            "train_cycle_min", "train_cycle_max", "test_rows", "test_target_column_loaded",
            "sample_rows", "test_ids_sha256", "sample_ids_sha256", "baseline_cache_sha256",
            "baseline_cache_route_reconstructed_bit_exact", "baseline_missing_mask_count",
            "baseline_present_mask_count", "feature_allowlist", "parameters_match_historical_source",
            "protocol_boundary",
        )
        if any(previous.get(key) != report.get(key) for key in stable_fields):
            raise RuntimeError("C19 preflight stable fields changed; frozen run cannot resume")
        # The first approved live resource snapshot remains immutable. Current
        # resources have still been checked above for every invocation.
        report = previous
    else:
        _write_once(preflight_path, report)
    return report


def _fit_events() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    events = read_events(LEDGER_PATH)
    start = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == FIT_ID]
    terminal = [event for event in events if event.get("fit_id") == FIT_ID
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(start) > 1 or len(terminal) > 1:
        raise RuntimeError("C19 fit ledger has duplicate start or terminal events")
    return start, terminal[0] if terminal else None


def _fit_or_resume(train: pd.DataFrame) -> tuple[C19FullTrainClassifier, dict[str, Any], dict[str, Any]]:
    _protocol, freeze = _load_frozen()
    model_path = RUN_ROOT / "bundle/model.joblib"
    state_path = RUN_ROOT / "bundle/fit_state.json"
    receipt_path = RUN_ROOT / "bundle/fit_receipt.json"
    started, terminal = _fit_events()
    if terminal is not None and terminal.get("event") == "FIT_FAILED":
        raise RuntimeError("C19 full-Train task fit is recorded failed; no retry is authorized")

    if model_path.exists():
        model = joblib.load(model_path)
        state = model.audit_state()
        if state_path.exists():
            if state != _read_json(state_path):
                raise RuntimeError("Saved C19 model state differs from its durable state receipt")
        else:
            if not started:
                raise RuntimeError("C19 model exists without a consumed fit event")
            _write_once(state_path, state)
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
        else:
            if not started:
                raise RuntimeError("C19 model exists without a consumed fit event")
            receipt = {
                "fit_id": FIT_ID,
                "freeze_signature": freeze["freeze_signature"],
                "model_sha256": _sha(model_path),
                "state_sha256": canonical_sha256(state),
                "fit_rows": int(state["fit_rows"]),
                "fit_cycle_min": int(state["fit_cycle_min"]),
                "fit_cycle_max": int(state["fit_cycle_max"]),
                "observed_peak_rss_bytes": None,
                "estimator_fit_count": 1,
                "preprocessing_fit_population": "official Classification Train cycles 47-79 only",
                "test_data_used_for_fit": False,
                "sample_submission_used_for_fit": False,
                "persistence_recovered_from_durable_model": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            _write_once(receipt_path, receipt)
        receipt_check = dict(receipt)
        recorded_receipt_sha = receipt_check.pop("receipt_sha256", None)
        if (state != _read_json(state_path)
                or receipt.get("model_sha256") != _sha(model_path)
                or receipt.get("state_sha256") != canonical_sha256(state)
                or recorded_receipt_sha != canonical_sha256(receipt_check)
                or state.get("fit_rows") != EXPECTED_TRAIN_ROWS
                or state.get("cutoff") != 79
                or state.get("test_data_used") is not False):
            raise RuntimeError("Saved C19 model state/receipt failed resume verification")
        if terminal is None:
            finish_fit(LEDGER_PATH, fit_id=FIT_ID, status="FIT_SAVED",
                       receipt_sha256=receipt["receipt_sha256"])
        return model, state, receipt

    if started:
        raise RuntimeError("C19 fit start exists without a durable fitted model; no refit/retry is authorized")
    if terminal is not None:
        raise RuntimeError("C19 terminal fit receipt exists without its durable fitted model")

    with threadpool_limits(limits=THREADS):
        model = C19FullTrainClassifier()
        if model.cutoff != 79:
            raise RuntimeError("C19 full-Train wrapper did not set cutoff 79")
        # Start is appended immediately before the single estimator fit.
        start_fit(LEDGER_PATH, fit_id=FIT_ID, card_id=CARD_ID, stage=STAGE,
                  cutoff=79, component="C19_XGB_SVD32_FULLTRAIN")
        try:
            with _PeakRSS() as monitor:
                model.fit(train.loc[:, list(FEATURE_NAMES)].copy(),
                          train["artifact_signal"].to_numpy(copy=True))
            state = model.audit_state()
            if state.get("fit_rows") != EXPECTED_TRAIN_ROWS or state.get("cutoff") != 79:
                raise RuntimeError("C19 fitted state does not match full-Train cutoff 79")
            if state.get("test_data_used") is not False or state.get("validation_labels_passed_to_fit") is not False:
                raise RuntimeError("C19 fitted state reports forbidden source usage")
            if monitor.peak > JOB_CAP_BYTES:
                raise RuntimeError(f"C19 observed peak RSS {monitor.peak} exceeded the 6 GiB cap")

            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_name("model.joblib.tmp")
            joblib.dump(model, temporary, compress=3)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, model_path)
            _write_once(state_path, state)
            receipt_value = {
                "fit_id": FIT_ID,
                "freeze_signature": freeze["freeze_signature"],
                "model_sha256": _sha(model_path),
                "state_sha256": canonical_sha256(state),
                "fit_rows": int(state["fit_rows"]),
                "fit_cycle_min": int(state["fit_cycle_min"]),
                "fit_cycle_max": int(state["fit_cycle_max"]),
                "observed_peak_rss_bytes": int(monitor.peak),
                "estimator_fit_count": 1,
                "preprocessing_fit_population": "official Classification Train cycles 47-79 only",
                "test_data_used_for_fit": False,
                "sample_submission_used_for_fit": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            receipt_value["receipt_sha256"] = canonical_sha256(receipt_value)
            _write_once(receipt_path, receipt_value)
            finish_fit(LEDGER_PATH, fit_id=FIT_ID, status="FIT_SAVED",
                       receipt_sha256=receipt_value["receipt_sha256"])
            return model, state, receipt_value
        except Exception as exc:
            # A fit has been consumed at FIT_STARTED.  It is never retried.
            current_start, current_terminal = _fit_events()
            if current_start and current_terminal is None and not model_path.exists():
                finish_fit(LEDGER_PATH, fit_id=FIT_ID, status="FIT_FAILED", error=type(exc).__name__)
            raise


def _write_predictions_once(
    model: C19FullTrainClassifier,
    state: dict[str, Any],
    receipt: dict[str, Any],
    test: pd.DataFrame,
    sample: pd.DataFrame,
    baseline: dict[str, np.ndarray],
) -> Path:
    prediction_path = RUN_ROOT / "predictions/test_predictions.npz"
    prediction_receipt_path = RUN_ROOT / "predictions/test_predictions.json"
    if prediction_path.exists() or prediction_receipt_path.exists():
        if not prediction_path.exists() or not prediction_receipt_path.exists():
            raise RuntimeError("C19 prediction output is incomplete; refusing to overwrite")
        receipt_existing = _read_json(prediction_receipt_path)
        if receipt_existing.get("npz_sha256") != _sha(prediction_path):
            raise RuntimeError("C19 prediction output hash differs from its receipt")
        return prediction_path

    with threadpool_limits(limits=THREADS):
        raw = model.predict_probability(test.loc[:, list(FEATURE_NAMES)].copy())
    mask = baseline["missing_mask"]
    current_probability = baseline["probability"]
    candidate_probability = _route_c19_probability(
        current_probability, raw, baseline["missing"], mask
    )
    labels = (candidate_probability >= THRESHOLD).astype(np.int8)
    if (candidate_probability.shape != (EXPECTED_TEST_ROWS,)
            or not np.isfinite(raw).all() or not np.isfinite(candidate_probability).all()
            or np.any((raw < 0) | (raw > 1))
            or np.any((candidate_probability < 0) | (candidate_probability > 1))
            or labels.dtype != np.int8
            or not np.array_equal(candidate_probability[mask], baseline["missing"][mask])):
        raise RuntimeError("C19 returned invalid probability, route, or label outputs")
    if sample["record_id"].astype(str).tolist() != test["record_id"].astype(str).tolist():
        raise RuntimeError("C19 Test and SampleSubmission IDs differ in order")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction_path.with_name("test_predictions.npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            record_ids=test["record_id"].astype(str).to_numpy(dtype=str),
            raw_probability=raw.astype(np.float64),
            baseline_probability=current_probability.astype(np.float64),
            baseline_missing=baseline["missing"].astype(np.float64),
            missing_mask=mask.astype(bool),
            candidate_probability=candidate_probability.astype(np.float64),
            labels=labels.astype(np.int8),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, prediction_path)
    value = {
        "fit_receipt_sha256": receipt["receipt_sha256"],
        "model_state_sha256": canonical_sha256(state),
        "npz_sha256": _sha(prediction_path),
        "test_ids_sha256": ordered_ids_sha256(test["record_id"].astype(str).tolist()),
        "raw_probability_sha256": hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest(),
        "candidate_probability_sha256": hashlib.sha256(candidate_probability.astype("<f8").tobytes()).hexdigest(),
        "labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
        "prediction_rows": int(len(labels)),
        "test_target_values_loaded": False,
        "sample_submission_used_for_prediction": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_once(prediction_receipt_path, value)
    return prediction_path


def _route_c19_probability(
    baseline_probability: np.ndarray, raw_probability: np.ndarray,
    missing_probability: np.ndarray, missing_mask: np.ndarray,
) -> np.ndarray:
    """Apply the frozen sparse25/C19 present route and exact missing fallback."""
    baseline = np.asarray(baseline_probability, dtype=np.float64)
    raw = np.asarray(raw_probability, dtype=np.float64)
    missing_p = np.asarray(missing_probability, dtype=np.float64)
    mask = np.asarray(missing_mask, dtype=bool)
    if (baseline.ndim != 1 or raw.shape != baseline.shape or missing_p.shape != baseline.shape
            or mask.shape != baseline.shape or not np.isfinite(baseline).all()
            or not np.isfinite(raw).all() or not np.isfinite(missing_p).all()
            or np.any((baseline < 0) | (baseline > 1))
            or np.any((raw < 0) | (raw > 1))
            or np.any((missing_p < 0) | (missing_p > 1))):
        raise ValueError("C19 route requires aligned finite probability vectors")
    result = np.where(mask, missing_p, (1.0 - BLEND_WEIGHT) * baseline + BLEND_WEIGHT * raw)
    if not np.array_equal(result[mask], missing_p[mask]):
        raise RuntimeError("C19 missing fallback changed cached probabilities")
    return result


def _prepare_csv(test: pd.DataFrame, sample: pd.DataFrame, prediction_path: Path,
                 state: dict[str, Any], fit_receipt: dict[str, Any],
                 preflight: dict[str, Any]) -> dict[str, Any]:
    csv_path = ROOT / "submissions/optimization_v3/c19_benchmark_v1/c19_xgb_svd32.csv"
    manifest_path = REPORT_ROOT / "submission_manifest.json"
    with np.load(prediction_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    required = {"record_ids", "raw_probability", "baseline_probability", "baseline_missing",
                "missing_mask", "candidate_probability", "labels"}
    if set(values) != required:
        raise RuntimeError("C19 saved prediction bundle schema changed")
    ids = test["record_id"].astype(str).tolist()
    if (values["record_ids"].astype(str).tolist() != ids
            or ids != sample["record_id"].astype(str).tolist()
            or len(set(ids)) != EXPECTED_TEST_ROWS):
        raise RuntimeError("C19 saved prediction IDs do not exactly match the official SampleSubmission")
    expected = np.where(
        values["missing_mask"],
        values["baseline_missing"],
        (1.0 - BLEND_WEIGHT) * values["baseline_probability"] + BLEND_WEIGHT * values["raw_probability"],
    )
    if (not np.array_equal(expected, values["candidate_probability"])
            or not np.array_equal((expected >= THRESHOLD).astype(np.int8), values["labels"])
            or not np.array_equal(values["candidate_probability"][values["missing_mask"]],
                                  values["baseline_missing"][values["missing_mask"]])):
        raise RuntimeError("C19 final labels do not reproduce the frozen route and threshold")
    if csv_path.exists():
        frame = pd.read_csv(csv_path, dtype={"record_id": str})
        if (frame.columns.tolist() != sample.columns.tolist()
                or frame["record_id"].astype(str).tolist() != ids
                or not np.array_equal(frame["artifact_signal"].to_numpy(dtype=np.int8), values["labels"])):
            raise RuntimeError("Existing C19 CSV differs; refusing overwrite")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = csv_path.with_name(csv_path.name + ".tmp")
        pd.DataFrame({"record_id": test["record_id"].to_numpy(),
                      "artifact_signal": values["labels"]}).to_csv(temporary, index=False)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, csv_path)
        frame = pd.read_csv(csv_path, dtype={"record_id": str})
    if (len(frame) != EXPECTED_TEST_ROWS or frame.columns.tolist() != sample.columns.tolist()
            or frame["record_id"].astype(str).tolist() != ids
            or frame["record_id"].isna().any() or frame["record_id"].duplicated().any()
            or frame["artifact_signal"].dtype.kind not in "iu"
            or not set(frame["artifact_signal"].unique()).issubset({0, 1})
            or not np.isfinite(frame["artifact_signal"].to_numpy(dtype=np.float64)).all()):
        raise RuntimeError("C19 submission CSV failed official-schema verification")

    result = {
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED",
        "candidate_id": "C19_XGB_SVD32_BENCHMARK",
        "competition": COMPETITION,
        "submission_message": SUBMISSION_MESSAGE,
        "historical_C19_verdict_remains": "RETIRE / prior frozen-gate failure; this full-Train upload is a separate benchmark exception",
        "source_commit": _read_json(FREEZE_PATH)["identity"]["source_commit"],
        "protocol_sha256": _read_json(FREEZE_PATH)["protocol_sha256"],
        "freeze_signature": _read_json(FREEZE_PATH)["freeze_signature"],
        "official_train_rows": EXPECTED_TRAIN_ROWS,
        "official_train_sha256": TRAIN_SHA256,
        "fit_cycle_range": [47, 79],
        "fit_predictor_count": 1,
        "fit_receipt_sha256": fit_receipt["receipt_sha256"],
        "fit_state_sha256": canonical_sha256(state),
        "model_sha256": fit_receipt["model_sha256"],
        "test_rows": EXPECTED_TEST_ROWS,
        "test_target_column_loaded": False,
        "preprocessing_fit_only_train": True,
        "test_features_used_for_inference_only": True,
        "sample_submission_used_for_schema_verification_only": True,
        "input_sha256": dict(_read_json(FREEZE_PATH)["input_sha256"]),
        "baseline_cache_sha256": _sha(BASE_CACHE),
        "baseline_manifest_sha256": _sha(BASE_MANIFEST),
        "baseline_csv_sha256": _sha(BASE_CSV),
        "route": {
            "present": ".75*current sparse25 probability + .25*C19 raw probability",
            "missing": "exact frozen sparse25 CB_MISSING probability",
            "threshold": THRESHOLD,
        },
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "csv_sha256": _sha(csv_path),
        "test_ids_sha256": ordered_ids_sha256(ids),
        "columns": frame.columns.tolist(),
        "rows": len(frame),
        "prediction_counts": {str(int(k)): int(v) for k, v in frame["artifact_signal"].value_counts().sort_index().items()},
        "candidate_labels_differing_from_sparse25": int(np.count_nonzero(values["labels"] != (values["baseline_probability"] >= 0.5).astype(np.int8))),
        "missing_branch_label_changes": int(np.count_nonzero(values["labels"][values["missing_mask"]] != (values["baseline_missing"][values["missing_mask"]] >= 0.5).astype(np.int8))),
        "kaggle_preflight_path": "reports/optimization_v3/c19_benchmark_v1/kaggle_preflight.json",
        "preflight_sha256": canonical_sha256(preflight),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "threadpoolctl", "joblib"]},
    }
    _write_once(manifest_path, result)
    return result


def run() -> dict[str, Any]:
    preflight = _preflight()
    train, test, sample = _load_train_test_sample()
    baseline = _load_baseline_cache(test, sample)
    model, state, receipt = _fit_or_resume(train)
    prediction_path = _write_predictions_once(model, state, receipt, test, sample, baseline)
    manifest = _prepare_csv(test, sample, prediction_path, state, receipt, preflight)
    verifier = subprocess.run(
        [sys.executable, "-m", "scripts.optimization_v3_c19_benchmark_verify"],
        cwd=ROOT, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if verifier.returncode != 0:
        raise RuntimeError("Independent C19 CSV verifier failed; fitted state is preserved and must not be refit")
    print(json.dumps({"status": manifest["status"], "csv_path": manifest["csv_path"],
                      "csv_sha256": manifest["csv_sha256"], "fit_id": FIT_ID}, ensure_ascii=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="verify frozen inputs without any model fit")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(_preflight(), ensure_ascii=False))
    else:
        run()


if __name__ == "__main__":
    main()
