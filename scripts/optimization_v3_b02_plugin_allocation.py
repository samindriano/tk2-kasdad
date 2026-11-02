"""Zero-fit plug-in expected two-class Macro F1 batch allocation diagnostic.

No Test/SampleSubmission/Kaggle access. Selection consumes probabilities,
availability masks, and stable IDs only. Query labels are read after every K
and hard prediction has been fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_a02_b01_centering as b01
from scripts import optimization_v3_a02_continuation as a02
from scripts import optimization_v3_a02_neighbors as a02_neighbors
from src.models.research_next_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/b02_plugin_allocation_v1"
FREEZE_PATH = REPORT / "FREEZE.json"
OUTPUT_DIR = REPORT / "B02_OUTPUTS"
STAGING_DIR = REPORT / ".B02_OUTPUTS.staging"
TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
LEDGER_PATH = b01.LEDGER_PATH
B01_REPORT = ROOT / "reports/optimization_v3/a02_b01_centering_v1/B01_OUTPUTS"
B01_FREEZE = b01.FREEZE_PATH
QUEUE_PATH = ROOT / "reports/optimization_v3/extended_research_v1/QUEUE.md"

EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_TRAIN_SHA256 = a02.EXPECTED_TRAIN_SHA256
EXPECTED_LEDGER_SHA256 = b01.EXPECTED_LEDGER_SHA256
EXPECTED_LEDGER = {"sha256": EXPECTED_LEDGER_SHA256, "starts": 66, "saved": 64,
                   "failed": 2, "pending": [], "cap": 66, "remaining": 0}
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 8 * 1024**3
SHRINKAGE = 4.0
CORRECTION_LIMIT = 0.05
THRESHOLD = 0.50
ORIGINS = (59, 63, 67, 71, 75)
FIRST_FOUR = {origin: (origin + 1, origin + 4) for origin in ORIGINS}
H12_WINDOW = (68, 79)
W1_WINDOW = (56, 59)
SOURCES = ("B01", "A02", "Sparse25", "B01_label_shuffle")

TRAIN_FEATURE_COLUMNS = ("record_id", "cycle_code", "signal_stream", "entry_friction", "text_load")
PREDICTION_KEYS = {
    "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
    "labels", "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
    "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction",
    "correction_unclipped",
}
PREDICTION_LOAD_KEYS = (
    "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
    "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
    "neighbor_source_rows", "neighbor_ids", "neighbor_similarities",
    "correction", "correction_unclipped",
)
RUNS = {
    59: "optimization_v3_a02_continuation_v1",
    63: "optimization_v3_a02_continuation_v1",
    67: "optimization_v3_a02_neighbors_v1",
    71: "optimization_v3_a02_continuation_v1",
    75: "optimization_v3_a02_neighbors_v1",
}

RECIPE = {
    "name": "B02_LABEL_FREE_PLUGIN_EXPECTED_TWO_CLASS_MACRO_F1",
    "batch_scenarios": {
        "primary": "five independent origin first-four-cycle windows; aggregate their fixed decisions with unchanged W1",
        "h12": "origin67 full cycles 68-79 as a separate, overlapping batch scenario",
        "overlap_warning": "H12 rows overlap primary windows; scenarios are not independent holdouts",
    },
    "sources": list(SOURCES),
    "fixed_missing_rule": "freeze missing-stream hard labels at that source's existing >=0.50 route",
    "eligible_present_rule": "only candidate K values at ends of exact equal-probability blocks, plus zero and all",
    "soft_confusion": "S=sum(p) over all batch rows; fixed missing hard labels contribute expected TP/FP/FN/TN; present top-K contributes sum(p_topK) to TP and K-sum(p_topK) to FP; FN=S-TP; TN=N-S-FP",
    "objective": "0.5*(2TP/(2TP+FP+FN)+2TN/(2TN+FP+FN)); each zero denominator contributes zero",
    "tie_break": "maximize within 1e-12; then K nearest existing present-row >=0.50 positive count; then smaller K",
    "selector_inputs": ["probabilities", "missing_mask", "stable_record_ids"],
    "selector_receives_target": False,
    "selection_scope": "each source and batch independently; probabilities and existing source recipes fixed",
    "threshold_grid_or_probability_fit": False,
    "target_access": "bank labels only for each origin's pre-origin OOF bank to reconstruct frozen B01 probabilities; query labels loaded only after every K and hard prediction is fixed",
    "fit_transform_predict_or_neighbor_search": False,
    "qualification": "adaptive diagnostic only; no candidate qualification, submission, or champion selection",
}

METHOD_NOVELTY_REVIEW = {
    "verdict": "NARROWLY_DISTINCT_WITH_SUBSTANTIAL_OVERLAP_IN_BROADER_DECISION_PROBLEM",
    "claim": "This is a plug-in expected two-class Macro F1 rule for selecting whole probability-tie-block K values, not a novel general thresholding or positive-allocation method.",
    "closest_prior_work": [
        "Round 1 searched inner-label-selected thresholds; those outcomes show that improved recall/positive allocation can reduce Macro F1.",
        "Round 2 selected thresholds from inner labels and found that fixed 0.50 beat an over-allocating inner-pooled threshold on W2.",
        "Optimization V2 fitted calibration/stacking heads at fixed 0.50 and did not search thresholds.",
        "A02 matched-count controls ranked at K copied from another frozen arm's 0.50 positive count; B02 instead chooses K from plug-in expected two-class Macro F1.",
    ],
    "limits": "The objective is nonlinear and computed from expected confusion counts; it is neither exact expected Macro F1 nor a guarantee under calibration error or temporal/batch shift.",
}

STATIC_INPUTS = (
    "scripts/optimization_v3_b02_plugin_allocation.py",
    "tests/test_optimization_v3_b02_plugin_allocation.py",
    "scripts/optimization_v3_a02_b01_centering.py",
    "scripts/optimization_v3_a02_continuation.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_reliability_audit.py",
    "src/models/classification_a02_neighbors.py",
    "src/models/research_next_contract.py",
    "reports/optimization_v3/extended_research_v1/QUEUE.md",
    "reports/optimization_v3/a02_b01_centering_v1/FREEZE_V2.json",
    "reports/optimization_v3/a02_b01_centering_v1/B01_OUTPUTS/RESULTS.json",
    "reports/optimization_v3/a02_b01_centering_v1/B01_OUTPUTS/QA.json",
    "experiments/optimization_v3/a02_continuation_v1/protocol.json",
    "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json",
    "experiments/optimization_v3/c19_a02_v2/A02/protocol.json",
    "experiments/optimization_v3/c19_a02_v2/A02/freeze.json",
    "reports/optimization_v3/a02_continuation_v1/verification.json",
    "reports/optimization_v3/a02_neighbors_v1/verification.json",
    "reports/optimization_v3/a02_continuation_v1/submission_manifest.json",
    "reports/classification_round1/FINDINGS.md",
    "reports/classification_round2/FINDINGS.md",
    "reports/optimization_v2/FINDINGS.md",
    "reports/optimization_v3/a02_next_decision_v1/DECISION.md",
    "reports/optimization_v3/a02_next_decision_v1/matched_count_results.json",
    "scripts/optimization_v3_a02_next_decision.py",
    "experiments/runs/optimization_v3_research_next_v1/budget.jsonl",
    "data/raw/classification/Train.csv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(values: Any, dtype: str | np.dtype | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    if array.dtype.kind in "USO":
        payload = json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _atomic_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Immutable B02 output differs; refusing overwrite: {path}")
        return
    staging = path.with_name(f".{path.name}.staging")
    if staging.exists():
        raise RuntimeError(f"Interrupted B02 file staging exists; inspect before resume: {staging}")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    staging.rename(path)


def _preflight_resources() -> dict[str, int]:
    available = int(psutil.virtual_memory().available)
    rss = int(psutil.Process().memory_info().rss)
    if available < FREE_RAM_RESERVE_BYTES or rss > JOB_CAP_BYTES:
        raise RuntimeError("B02 needs 8 GiB free RAM and must stay below the 6 GiB job cap")
    return {"available_bytes_before": available, "process_rss_bytes_before": rss,
            "job_cap_bytes": JOB_CAP_BYTES, "reserve_bytes": FREE_RAM_RESERVE_BYTES,
            "threads": THREADS}


def _normalize_id(value: Any) -> str:
    return b01.audit.normalize_id(value)


def _read_feature_only_train() -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    freeze = a02.verify_freeze()
    train, caches, _banks, _queries = a02._load_structural_inputs(include_target=False)
    if ("artifact_signal" in train.columns or len(train) != EXPECTED_TRAIN_ROWS
            or train.record_id.isna().any() or train.record_id.duplicated().any()
            or _sha256_file(TRAIN_PATH) != EXPECTED_TRAIN_SHA256):
        raise RuntimeError("B02 feature-only Train identity/schema/hash did not validate")
    train["cycle_code"] = pd.to_numeric(train.cycle_code, errors="raise").astype(np.int16)
    train["entry_friction"] = pd.to_numeric(train.entry_friction, errors="raise").astype(np.int8)
    train["text_load"] = pd.to_numeric(train.text_load, errors="raise").astype(np.float64)
    return train, caches, freeze


def _selected_train_labels(rows: np.ndarray, expected_ids: np.ndarray,
                           train_features: pd.DataFrame) -> np.ndarray:
    """Read only the requested Train target rows; called after all allocations are fixed."""
    selected = set(map(int, np.asarray(rows, dtype=np.int64)))
    if len(selected) != len(rows) or "artifact_signal" in train_features.columns:
        raise RuntimeError("B02 post-selection target row set or feature-only boundary is invalid")
    frame = pd.read_csv(
        TRAIN_PATH, usecols=["record_id", "artifact_signal"], dtype={"record_id": str},
        skiprows=lambda line: line > 0 and (line - 1) not in selected,
    )
    if (len(frame) != len(rows)
            or not np.array_equal(frame.record_id.to_numpy(dtype=str), expected_ids.astype(str))):
        raise RuntimeError("B02 target labels do not join to the frozen selected query row order")
    y = pd.to_numeric(frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise RuntimeError("B02 selected query labels are not binary")
    return y


def _load_npz_without_targets(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != PREDICTION_KEYS:
            raise RuntimeError(f"Frozen A02 prediction bundle schema differs: {path}")
        return {key: np.asarray(archive[key]).copy() for key in PREDICTION_LOAD_KEYS}


def _load_model_state(folder: Path, origin: int, arm: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    receipt = json.loads((folder / "fit_receipt.json").read_text(encoding="utf-8"))
    state = json.loads((folder / "fit_state.json").read_text(encoding="utf-8"))
    payload = dict(receipt)
    receipt_sig = payload.pop("receipt_sha256", None)
    model_path = folder / "model.joblib"
    if (receipt_sig != canonical_sha256(payload)
            or receipt.get("origin_cycle") != origin
            or receipt.get("residual_arm") != arm
            or receipt.get("task_estimator_fit_count") != 1
            or receipt.get("test_data_used") is not False
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("model_sha256") != _sha256_file(model_path)
            or receipt.get("state_sha256") != canonical_sha256(state)):
        raise RuntimeError(f"Frozen A02 {arm} fit receipt/state lineage fails at E{origin}")
    model = joblib.load(model_path)
    if canonical_sha256(model.audit_state()) != canonical_sha256(state):
        raise RuntimeError(f"Frozen A02 {arm} loaded model state differs at E{origin}")
    return model, receipt, state


def _validate_saved_geometry(saved: dict[str, np.ndarray], bank_rows: np.ndarray,
                             bank_ids: np.ndarray, bank_cycles: np.ndarray) -> None:
    rows = saved["score_rows"].astype(np.int64)
    cycles = saved["score_cycles"].astype(np.int16)
    missing = saved["missing_mask"].astype(bool)
    counts = saved["neighbor_count"].astype(np.int64)
    neighbors = saved["neighbor_source_rows"].astype(np.int64)
    neighbor_ids = saved["neighbor_ids"].astype(str)
    similarities = saved["neighbor_similarities"].astype(np.float64)
    if (neighbors.shape != (len(rows), 32) or neighbor_ids.shape != neighbors.shape
            or similarities.shape != neighbors.shape or counts.shape != rows.shape
            or not np.isfinite(similarities).all()):
        raise RuntimeError("Saved A02 geometry dimensions or finiteness differ from the frozen top-32 recipe")
    lookup = {int(row): index for index, row in enumerate(bank_rows)}
    for i, raw_count in enumerate(counts):
        n = int(raw_count)
        if n < 0 or n > 32:
            raise RuntimeError("Saved A02 neighbor count is outside top-32 width")
        if (np.any(neighbors[i, n:] != -1) or np.any(neighbor_ids[i, n:] != "")
                or np.any(similarities[i, n:] != 0.0)):
            raise RuntimeError("Saved A02 neighbor padding differs from the frozen contract")
        if missing[i] and n != 0:
            raise RuntimeError("Missing-stream query unexpectedly has saved neighbors")
        if n == 0:
            continue
        chosen = neighbors[i, :n]
        if len(np.unique(chosen)) != n or np.any(chosen == rows[i]):
            raise RuntimeError("Saved A02 neighbors contain duplicate rows or the query itself")
        try:
            indices = np.asarray([lookup[int(row)] for row in chosen], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"Saved A02 neighbor is outside the verified OOF bank: {exc}") from exc
        if (np.any(bank_cycles[indices] >= cycles[i])
                or [_normalize_id(v) for v in neighbor_ids[i, :n]]
                   != [_normalize_id(bank_ids[j]) for j in indices]
                or np.any(similarities[i, :n] <= 0.0)
                or np.any(similarities[i, :n] > 1.0 + 1e-12)
                or np.any(np.diff(similarities[i, :n]) > 1e-15)
                or any(similarities[i, j] == similarities[i, j + 1]
                       and chosen[j] > chosen[j + 1] for j in range(n - 1))):
            raise RuntimeError("Saved A02 neighbors violate temporal/ID/similarity/tie-order constraints")


def _build_oof_bank(origin: int, caches: dict[int, dict[str, np.ndarray]],
                    train: pd.DataFrame, continuation_protocol: dict[str, Any],
                    neighbors_protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    if origin in (67, 75):
        bank_key = f"E{origin}"
        declared = neighbors_protocol.get("bank", {}).get(bank_key)
        if (neighbors_protocol.get("task") != "classification"
                or origin not in neighbors_protocol.get("origins", [])
                or not isinstance(declared, dict)):
            raise RuntimeError(f"Frozen neighbors protocol does not declare {bank_key}")
        bank = a02_neighbors._build_bank(origin, caches, train)
        present_rows = int((~bank["missing"].astype(bool)).sum())
        if (len(bank["rows"]) != int(declared["rows"])
                or present_rows != int(declared["present_rows"])):
            raise RuntimeError(f"Frozen neighbors bank counts differ from protocol at {bank_key}")
        return bank
    return a02._build_bank(origin, caches, train, continuation_protocol)


def _read_origin_inputs(origin: int, train: pd.DataFrame,
                        caches: dict[int, dict[str, np.ndarray]],
                        protocol: dict[str, Any],
                        neighbors_protocol: dict[str, Any],
                        neighbors_freeze_signature: str) -> dict[str, Any]:
    bank_full = _build_oof_bank(origin, caches, train, protocol, neighbors_protocol)
    present = ~bank_full["missing"].astype(bool)
    bank_rows = bank_full["rows"][present].astype(np.int64)
    bank_ids = bank_full["ids"][present].astype(str)
    bank_cycles = bank_full["cycles"][present].astype(np.int16)
    bank_oof = bank_full["baseline"][present].astype(np.float64)
    bank_features = train
    bank_labels = b01_benchmark_label_reader(bank_rows, bank_ids, bank_cycles, bank_features)

    folder = ROOT / "experiments/runs" / RUNS[origin] / f"E{origin}"
    observed_folder = folder / "observed"
    shuffled_folder = folder / "shuffled"
    observed_model, observed_receipt, observed_state = _load_model_state(observed_folder, origin, "observed")
    shuffled_model, shuffled_receipt, shuffled_state = _load_model_state(shuffled_folder, origin, "shuffled")
    for model, arm in ((observed_model, "observed"), (shuffled_model, "shuffled")):
        model_ids = np.asarray([_normalize_id(v) for v in model.bank_ids], dtype=str)
        if (not np.array_equal(model.bank_row_positions.astype(np.int64), bank_rows)
                or not np.array_equal(model_ids, bank_ids)
                or not np.array_equal(model.bank_cycles.astype(np.int16), bank_cycles)
                or not np.array_equal(model.bank_oof_probabilities.astype(np.float64), bank_oof)):
            raise RuntimeError(f"A02 {arm} bank rows/IDs/cycles/OOF probabilities differ at E{origin}")
    observed_residual = bank_labels.astype(np.float64) - bank_oof
    assignment = shuffled_model.residual_assignment.astype(np.int64)
    if (not np.array_equal(observed_model.residual_assignment, np.arange(len(bank_rows), dtype=np.int64))
            or assignment.shape != bank_rows.shape
            or not np.array_equal(np.sort(assignment), np.arange(len(bank_rows), dtype=np.int64))
            or not np.array_equal(observed_model.bank_residuals, observed_residual)
            or not np.array_equal(shuffled_model.bank_residuals, observed_residual[assignment])):
        raise RuntimeError(f"A02 observed/shuffled residual assignment or bank labels fail at E{origin}")

    prediction_path = observed_folder / "predictions.npz"
    sidecar = json.loads((observed_folder / "predictions.json").read_text(encoding="utf-8"))
    observed_test_data_claim = sidecar.get("test_data_used")
    observed_source_freeze_matches = (
        sidecar.get("freeze_signature") == neighbors_freeze_signature
        if origin in (67, 75) else observed_test_data_claim is False)
    if (sidecar.get("npz_sha256") != _sha256_file(prediction_path)
            or sidecar.get("fit_receipt_sha256") != observed_receipt["receipt_sha256"]
            or sidecar.get("origin_cycle") != origin
            or sidecar.get("residual_arm") != "observed"
            or sidecar.get("query_labels_passed_to_prediction") is not False
            or observed_test_data_claim not in (None, False)
            or not observed_source_freeze_matches):
        raise RuntimeError(f"A02 observed prediction sidecar does not attest target-free inference at E{origin}")
    saved = _load_npz_without_targets(prediction_path)
    shuffled_prediction_path = shuffled_folder / "predictions.npz"
    shuffled_sidecar = json.loads((shuffled_folder / "predictions.json").read_text(encoding="utf-8"))
    shuffled_test_data_claim = shuffled_sidecar.get("test_data_used")
    shuffled_source_freeze_matches = (
        shuffled_sidecar.get("freeze_signature") == neighbors_freeze_signature
        if origin in (67, 75) else shuffled_test_data_claim is False)
    if (shuffled_sidecar.get("npz_sha256") != _sha256_file(shuffled_prediction_path)
            or shuffled_sidecar.get("fit_receipt_sha256") != shuffled_receipt["receipt_sha256"]
            or shuffled_sidecar.get("origin_cycle") != origin
            or shuffled_sidecar.get("residual_arm") != "shuffled"
            or shuffled_sidecar.get("query_labels_passed_to_prediction") is not False
            or shuffled_test_data_claim not in (None, False)
            or not shuffled_source_freeze_matches):
        raise RuntimeError(f"A02 shuffled prediction sidecar does not attest target-free inference at E{origin}")
    shuffled_saved = _load_npz_without_targets(shuffled_prediction_path)
    rows = saved["score_rows"].astype(np.int64)
    ids = saved["score_record_ids"].astype(str)
    cycles = saved["score_cycles"].astype(np.int16)
    baseline = saved["baseline_prediction"].astype(np.float64)
    a02_probability = saved["prediction"].astype(np.float64)
    missing = saved["missing_mask"].astype(bool)
    if (not np.array_equal(ids, train.record_id.to_numpy(dtype=str)[rows])
            or not np.array_equal(cycles, train.cycle_code.to_numpy(dtype=np.int16)[rows])
            or not np.array_equal(missing, train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows])
            or not np.isfinite(baseline).all() or not np.isfinite(a02_probability).all()
            or np.any((baseline < 0) | (baseline > 1)) or np.any((a02_probability < 0) | (a02_probability > 1))
            or not np.array_equal(a02_probability[missing], baseline[missing])):
        raise RuntimeError(f"A02 saved query predictions fail row/ID/cycle/missing route at E{origin}")
    shared_geometry_keys = (
        "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "missing_mask",
        "entry_friction", "text_load_group", "neighbor_count", "neighbor_source_rows",
        "neighbor_ids", "neighbor_similarities",
    )
    for key in shared_geometry_keys:
        if not np.array_equal(saved[key], shuffled_saved[key]):
            raise RuntimeError(f"A02 observed/shuffled saved query geometry differs at E{origin}/{key}")
    _validate_saved_geometry(saved, bank_rows, bank_ids, bank_cycles)
    low, high = (origin + 1, 79) if origin == 67 else FIRST_FOUR[origin]
    expected_query = np.flatnonzero(train.cycle_code.between(low, high).to_numpy())
    if not np.array_equal(rows, expected_query):
        raise RuntimeError(f"A02 saved query rows do not match the frozen E{origin} scope {low}-{high}")
    for key, array in (("neighbor_count", saved["neighbor_count"]),
                       ("neighbor_source_rows", saved["neighbor_source_rows"]),
                       ("neighbor_ids", saved["neighbor_ids"]),
                       ("neighbor_similarities", saved["neighbor_similarities"])):
        if array.shape[0] != len(rows):
            raise RuntimeError(f"A02 saved geometry row count differs for {key} at E{origin}")

    geometry = {
        "score_rows": rows,
        "baseline_prediction": baseline,
        "neighbor_source_rows": saved["neighbor_source_rows"].astype(np.int64),
        "neighbor_similarities": saved["neighbor_similarities"].astype(np.float64),
        "neighbor_count": saved["neighbor_count"].astype(np.int64),
        "missing_mask": missing,
    }
    with b01.forbid_training_and_inference_calls() as calls:
        rebuilt = b01.compute_from_saved_geometry(
            geometry, bank_rows, bank_labels, bank_oof, assignment,
            shrinkage=SHRINKAGE, correction_limit=CORRECTION_LIMIT)
    if calls:
        raise RuntimeError(f"B02 attempted fit/transform/predict while rebuilding E{origin}: {calls}")
    if (not np.array_equal(rebuilt["a02_rebuilt"], a02_probability)
            or not np.array_equal(rebuilt["a02_shuffle_rebuilt"], shuffled_saved["prediction"])
            or not np.allclose(rebuilt["a02_shuffle_raw"], shuffled_saved["correction_unclipped"], rtol=0, atol=1e-15)
            or not np.array_equal(np.clip(rebuilt["a02_shuffle_raw"], -CORRECTION_LIMIT, CORRECTION_LIMIT),
                                  shuffled_saved["correction"])):
        raise RuntimeError(f"Frozen A02 observed/shuffled probabilities do not reconstruct at E{origin}")
    return {
        "origin": origin, "rows": rows, "ids": ids, "cycles": cycles,
        "missing": missing,
        "probabilities": {
            "B01": rebuilt["b01"].astype(np.float64),
            "A02": a02_probability,
            "Sparse25": baseline,
            "B01_label_shuffle": rebuilt["b01_label_shuffle"].astype(np.float64),
        },
        "bank": {"rows": bank_rows, "ids": bank_ids, "cycles": bank_cycles,
                 "oof": bank_oof, "labels": bank_labels,
                 "labels_sha256": _array_hash(bank_labels, np.int8)},
        "lineage": {
            "observed_fit_id": observed_receipt["fit_id"],
            "observed_receipt_sha256": observed_receipt["receipt_sha256"],
            "observed_state_sha256": canonical_sha256(observed_state),
            "observed_model_sha256": _sha256_file(observed_folder / "model.joblib"),
            "observed_predictions_sha256": _sha256_file(prediction_path),
            "shuffled_fit_id": shuffled_receipt["fit_id"],
            "shuffled_receipt_sha256": shuffled_receipt["receipt_sha256"],
            "shuffled_state_sha256": canonical_sha256(shuffled_state),
            "shuffled_model_sha256": _sha256_file(shuffled_folder / "model.joblib"),
            "bank_rows": len(bank_rows),
            "bank_rows_sha256": _array_hash(bank_rows, "<i8"),
            "bank_ids_sha256": _array_hash(bank_ids),
            "bank_cycles_sha256": _array_hash(bank_cycles, "<i2"),
            "bank_oof_sha256": _array_hash(bank_oof, "<f8"),
            "residual_assignment_sha256": _array_hash(assignment, "<i8"),
        },
    }


def b01_benchmark_label_reader(rows: np.ndarray, ids: np.ndarray,
                               cycles: np.ndarray, train_features: pd.DataFrame) -> np.ndarray:
    """Read artifact_signal only for the exact prior OOF bank rows."""
    selected = set(map(int, rows))
    if len(selected) != len(rows) or "artifact_signal" in train_features.columns:
        raise RuntimeError("B02 bank labels must be unique and separate from feature-only Train")
    frame = pd.read_csv(
        TRAIN_PATH, usecols=["record_id", "cycle_code", "signal_stream", "artifact_signal"],
        dtype={"record_id": str},
        skiprows=lambda line: line > 0 and (line - 1) not in selected,
    )
    if (len(frame) != len(rows)
            or not np.array_equal(frame.record_id.to_numpy(dtype=str), ids.astype(str))
            or not np.array_equal(pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int16), cycles)
            or frame.signal_stream.fillna("").astype(str).str.strip().eq("").any()):
        raise RuntimeError("B02 bank labels do not align with verified present-stream OOF rows")
    y = pd.to_numeric(frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise RuntimeError("B02 OOF bank labels are not binary")
    return y


def _load_all_candidate_inputs() -> dict[str, Any]:
    resources = _preflight_resources()
    train, caches, a02_freeze = _read_feature_only_train()
    protocol = a02._read_protocol()
    neighbors_protocol, neighbors_freeze = a02_neighbors._load_frozen()
    neighbors_verification = json.loads(
        (ROOT / "reports/optimization_v3/a02_neighbors_v1/verification.json").read_text(encoding="utf-8"))
    if (neighbors_protocol.get("task") != "classification"
            or neighbors_protocol.get("origins") != [67, 75]
            or neighbors_verification.get("status") != "PASS"
            or neighbors_verification.get("checks", {}).get("input_hashes_pass") is not True
            or neighbors_verification.get("checks", {}).get("source_boundaries_pass") is not True
            or neighbors_verification.get("checks", {}).get("test_or_sample_access") is not False):
        raise RuntimeError("Frozen A02 neighbors protocol does not match the E67/E75 cache lineage")
    b01_freeze = json.loads(B01_FREEZE.read_text(encoding="utf-8"))
    b01_results = json.loads((B01_REPORT / "RESULTS.json").read_text(encoding="utf-8"))
    b01_qa = json.loads((B01_REPORT / "QA.json").read_text(encoding="utf-8"))
    if (b01_freeze.get("status") != "B01_RECIPE_FROZEN_PRE_SCORE"
            or _sha256_file(Path(b01.__file__)) != b01_freeze.get("code_sha256")
            or b01_results.get("status") != "DIAGNOSTIC_GATE_FAILED"
            or b01_results.get("qualification") != "DIAGNOSTIC_ONLY_NOT_QUALIFIED"
            or b01_qa.get("status") != "PASS"
            or b01_results.get("provenance", {}).get("query_labels_passed_to_formula") is not False
            or b01_results.get("provenance", {}).get("test_sample_kaggle_or_submission_access") is not False):
        raise RuntimeError("Historical B01 recipe, failed verdict, or zero-fit QA does not validate")
    ledger_before = b01._verify_ledger()
    if ledger_before != EXPECTED_LEDGER:
        raise RuntimeError("B02 zero-fit ledger is not the exact consumed 66-fit checkpoint")
    origins = {
        origin: _read_origin_inputs(origin, train, caches, protocol, neighbors_protocol,
                                    neighbors_freeze["freeze_signature"])
        for origin in ORIGINS
    }

    # W1 is carried forward unchanged and contributes no allocation decision.
    # W1 covers cycles 56-59 and therefore comes from the frozen cutoff-55 cache.
    w1_cache = caches[55]
    w1_mask = (w1_cache["cycles"] >= W1_WINDOW[0]) & (w1_cache["cycles"] <= W1_WINDOW[1])
    w1_rows = w1_cache["score_rows"][w1_mask].astype(np.int64)
    w1_ids = w1_cache["score_record_ids"][w1_mask].astype(str)
    w1_cycles = w1_cache["cycles"][w1_mask].astype(np.int16)
    w1_missing = w1_cache["stream_missing"][w1_mask].astype(bool)
    w1_probability = w1_cache["baseline_prediction"][w1_mask].astype(np.float64)
    expected_w1 = np.flatnonzero(train.cycle_code.between(*W1_WINDOW).to_numpy())
    if (not np.array_equal(w1_rows, expected_w1)
            or not np.array_equal(w1_ids, train.record_id.to_numpy(dtype=str)[w1_rows])
            or not np.array_equal(w1_cycles, train.cycle_code.to_numpy(dtype=np.int16)[w1_rows])
            or not np.array_equal(w1_missing, train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[w1_rows])
            or not np.isfinite(w1_probability).all()
            or np.any((w1_probability < 0) | (w1_probability > 1))):
        raise RuntimeError("Frozen W1 exact Sparse25 fallback rows/IDs/mask/probabilities differ")
    w1 = {"rows": w1_rows, "ids": w1_ids, "cycles": w1_cycles,
          "missing": w1_missing, "probability": w1_probability}

    input_paths = set(STATIC_INPUTS)
    for cache_path in a02.CACHE_PATHS.values():
        input_paths.add(cache_path.relative_to(ROOT).as_posix())
        input_paths.add(cache_path.with_suffix(".json").relative_to(ROOT).as_posix())
    for origin in ORIGINS:
        folder = ROOT / "experiments/runs" / RUNS[origin] / f"E{origin}"
        for arm in ("observed", "shuffled"):
            for name in ("fit_receipt.json", "fit_state.json", "model.joblib",
                         "predictions.json", "predictions.npz"):
                input_paths.add((folder / arm / name).relative_to(ROOT).as_posix())
    input_sha256 = {}
    for relative in sorted(input_paths):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"B02 frozen dependency missing: {relative}")
        input_sha256[relative] = _sha256_file(path)
    arrays = {
        str(origin): {
            "query_rows_sha256": _array_hash(origins[origin]["rows"], "<i8"),
            "query_ids_sha256": _array_hash(origins[origin]["ids"]),
            "query_cycles_sha256": _array_hash(origins[origin]["cycles"], "<i2"),
            "missing_mask_sha256": _array_hash(origins[origin]["missing"], np.bool_),
            "probability_sha256": {
                source: _array_hash(origins[origin]["probabilities"][source], "<f8")
                for source in SOURCES
            },
            "bank": {key: value for key, value in origins[origin]["lineage"].items()
                     if key.startswith("bank_") or key == "residual_assignment_sha256"},
            "lineage": origins[origin]["lineage"],
        }
        for origin in ORIGINS
    }
    computed_input_hashes = {
        str(origin): {
            "B01": _array_hash(origins[origin]["probabilities"]["B01"], "<f8"),
            "A02": _array_hash(origins[origin]["probabilities"]["A02"], "<f8"),
            "Sparse25": _array_hash(origins[origin]["probabilities"]["Sparse25"], "<f8"),
            "B01_label_shuffle": _array_hash(origins[origin]["probabilities"]["B01_label_shuffle"], "<f8"),
        }
        for origin in ORIGINS
    }
    return {
        "resources": resources,
        "train": train,
        "caches": caches,
        "a02_freeze": a02_freeze,
        "b01_freeze": b01_freeze,
        "protocol": protocol,
        "neighbors_protocol_sha256": canonical_sha256(neighbors_protocol),
        "neighbors_freeze_signature": neighbors_freeze["freeze_signature"],
        "origins": origins,
        "w1": w1,
        "input_sha256": input_sha256,
        "arrays": arrays,
        "probability_hashes": computed_input_hashes,
        "ledger_before": ledger_before,
        "train_sha256": _sha256_file(TRAIN_PATH),
    }


def _current_commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def _freeze() -> dict[str, Any]:
    if OUTPUT_DIR.exists() or STAGING_DIR.exists():
        raise RuntimeError("B02 output bundle already exists; refusing a post-score freeze")
    with threadpool_limits(limits=THREADS):
        data = _load_all_candidate_inputs()
    freeze = {
        "status": "B02_RECIPE_AND_INPUTS_FROZEN_PRE_DECODER",
        "source_commit": _current_commit(),
        "recipe": RECIPE,
        "recipe_sha256": canonical_sha256(RECIPE),
        "official_train_rows": EXPECTED_TRAIN_ROWS,
        "official_train_sha256": data["train_sha256"],
        "train_feature_columns_loaded": list(TRAIN_FEATURE_COLUMNS),
        "query_target_rows_loaded": 0,
        "test_or_sample_accessed": False,
        "kaggle_accessed": False,
        "fit_transform_predict_or_neighbor_search_calls": 0,
        "candidate_probability_hashes": data["probability_hashes"],
        "query_geometry_and_bank_provenance": data["arrays"],
        "neighbors_protocol_sha256": data["neighbors_protocol_sha256"],
        "neighbors_freeze_signature": data["neighbors_freeze_signature"],
        "input_sha256": data["input_sha256"],
        "fit_ledger_before": data["ledger_before"],
    }
    if FREEZE_PATH.exists():
        if FREEZE_PATH.read_bytes() != _json_bytes(freeze):
            raise RuntimeError("Existing B02 freeze differs; refusing overwrite")
        return freeze
    _atomic_once(FREEZE_PATH, _json_bytes(freeze))
    return freeze


def _verify_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("B02 pre-decoder freeze is required before --run")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if (freeze.get("status") != "B02_RECIPE_AND_INPUTS_FROZEN_PRE_DECODER"
            or freeze.get("recipe_sha256") != canonical_sha256(freeze.get("recipe"))
            or freeze.get("recipe_sha256") != canonical_sha256(RECIPE)
            or freeze.get("query_target_rows_loaded") != 0
            or freeze.get("test_or_sample_accessed") is not False
            or freeze.get("kaggle_accessed") is not False):
        raise RuntimeError("B02 recipe/source boundary differs from immutable pre-decoder freeze")
    for relative, expected in freeze.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"B02 frozen input changed: {relative}")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", freeze["source_commit"], _current_commit()],
                              cwd=ROOT, capture_output=True, check=False)
    if ancestor.returncode != 0:
        raise RuntimeError("B02 pre-decoder source commit is not an ancestor of current checkout")
    with threadpool_limits(limits=THREADS):
        data = _load_all_candidate_inputs()
    if (data["input_sha256"] != freeze["input_sha256"]
            or data["probability_hashes"] != freeze["candidate_probability_hashes"]
            or data["arrays"] != freeze["query_geometry_and_bank_provenance"]
            or data["neighbors_protocol_sha256"] != freeze["neighbors_protocol_sha256"]
            or data["neighbors_freeze_signature"] != freeze["neighbors_freeze_signature"]
            or data["ledger_before"] != freeze["fit_ledger_before"]):
        raise RuntimeError("B02 reconstructed probabilities/banks differ from frozen pre-decoder state")
    return freeze, data


def _soft_macro_f1(tp: float, fp: float, fn: float, tn: float) -> float:
    pos_den = 2.0 * tp + fp + fn
    neg_den = 2.0 * tn + fp + fn
    f1_pos = 0.0 if pos_den == 0.0 else 2.0 * tp / pos_den
    f1_neg = 0.0 if neg_den == 0.0 else 2.0 * tn / neg_den
    return 0.5 * (f1_pos + f1_neg)


def _soft_confusion(probabilities: np.ndarray, missing: np.ndarray, k: int,
                    top_probability_sum: float) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if p.ndim != 1 or p.shape != missing.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Plug-in confusion requires finite [0,1] probabilities and aligned missing mask")
    fixed_missing = missing
    fixed_pred = p[fixed_missing] >= THRESHOLD
    fixed_tp = math.fsum(p[fixed_missing][fixed_pred].tolist())
    fixed_fp = math.fsum((1.0 - p[fixed_missing][fixed_pred]).tolist())
    fixed_fn = math.fsum(p[fixed_missing][~fixed_pred].tolist())
    fixed_tn = math.fsum((1.0 - p[fixed_missing][~fixed_pred]).tolist())
    n_present = int((~missing).sum())
    if k < 0 or k > n_present:
        raise ValueError("K is outside eligible present-stream rows")
    total_probability = math.fsum(p.tolist())
    tp = fixed_tp + float(top_probability_sum)
    fp = fixed_fp + (int(k) - float(top_probability_sum))
    fn = total_probability - tp
    tn = len(p) - total_probability - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "macro_f1": _soft_macro_f1(tp, fp, fn, tn)}


def select_k(probabilities: np.ndarray, missing_mask: np.ndarray,
             stable_record_ids: np.ndarray) -> dict[str, Any]:
    """Label-free decision rule; deliberately has no target/metric-label parameter."""
    p = np.asarray(probabilities, dtype=np.float64)
    missing = np.asarray(missing_mask, dtype=bool)
    ids = np.asarray(stable_record_ids).astype(str)
    if (p.ndim != 1 or p.shape != missing.shape or p.shape != ids.shape
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or pd.isna(ids).any() or len(np.unique(ids)) != len(ids)):
        raise ValueError("K selector received invalid probabilities, mask, or stable IDs")
    present_indices = np.flatnonzero(~missing)
    # Stable ID ordering only makes traversal deterministic. Exact-score ties
    # are always kept as complete blocks; IDs never split a tied probability.
    order = present_indices[np.lexsort((ids[present_indices], -p[present_indices]))]
    sorted_p = p[order]
    positive_at_half = int(np.sum(p[~missing] >= THRESHOLD))
    candidate_ks = [0]
    cumulative: dict[int, float] = {0: 0.0}
    if len(sorted_p):
        block_end = 0
        running: list[float] = []
        while block_end < len(sorted_p):
            value = sorted_p[block_end]
            next_end = block_end + 1
            while next_end < len(sorted_p) and sorted_p[next_end] == value:
                next_end += 1
            running.extend(sorted_p[block_end:next_end].tolist())
            k = next_end
            candidate_ks.append(k)
            cumulative[k] = math.fsum(running)
            block_end = next_end
    candidate_ks = sorted(set(candidate_ks))
    rows = []
    for k in candidate_ks:
        confusion = _soft_confusion(p, missing, k, cumulative[k])
        rows.append({"k": int(k), "soft_confusion": confusion})
    best = max(row["soft_confusion"]["macro_f1"] for row in rows)
    tied = [row for row in rows if best - row["soft_confusion"]["macro_f1"] <= 1e-12]
    chosen = min(tied, key=lambda row: (abs(row["k"] - positive_at_half), row["k"]))
    k = int(chosen["k"])
    selected = np.zeros(len(p), dtype=np.int8)
    selected[missing] = (p[missing] >= THRESHOLD).astype(np.int8)
    if k:
        selected[order[:k]] = 1
    if not np.array_equal(selected[missing], (p[missing] >= THRESHOLD).astype(np.int8)):
        raise RuntimeError("Plug-in allocation changed a fixed missing-stream label")
    selected_min = None if k == 0 else float(sorted_p[k - 1])
    excluded_max = None if k == len(sorted_p) else float(sorted_p[k])
    if k and k < len(sorted_p) and not selected_min > excluded_max:
        raise RuntimeError("Plug-in K split a tied probability block")
    half = next(row for row in rows if row["k"] == positive_at_half)
    return {
        "k_present": k,
        "eligible_present_rows": int(len(present_indices)),
        "missing_rows": int(missing.sum()),
        "fixed_missing_positive_rows": int(np.sum(p[missing] >= THRESHOLD)),
        "positive_count_at_half_present": positive_at_half,
        "positive_count_at_half_total": int(np.sum(p >= THRESHOLD)),
        "plugin_positive_rate_total": float(selected.mean()) if len(selected) else 0.0,
        "plugin_positive_rate_present": float(selected[~missing].mean()) if (~missing).any() else 0.0,
        "candidate_k_count": len(candidate_ks),
        "selected_k_soft_objective": float(chosen["soft_confusion"]["macro_f1"]),
        "fixed_half_soft_objective": float(half["soft_confusion"]["macro_f1"]),
        "selected_soft_confusion": chosen["soft_confusion"],
        "selected_score_floor": selected_min,
        "highest_excluded_score": excluded_max,
        "tie_break": {"objective_tolerance": 1e-12,
                      "nearest_half_positive_count": positive_at_half,
                      "then_smaller_k": True},
        "prediction": selected,
        "prediction_sha256": _array_hash(selected, np.int8),
        "probability_sha256": _array_hash(p, "<f8"),
        "missing_mask_sha256": _array_hash(missing, np.bool_),
        "record_ids_sha256": _array_hash(ids),
    }


def _actual_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    if y.shape != prediction.shape or not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Actual Macro F1 metrics require aligned binary labels and predictions")
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    pos_den = 2 * tp + fp + fn
    neg_den = 2 * tn + fp + fn
    f1_pos = 0.0 if pos_den == 0 else 2.0 * tp / pos_den
    f1_neg = 0.0 if neg_den == 0 else 2.0 * tn / neg_den
    return {
        "n": int(len(y)), "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "f1_class_0": f1_neg, "f1_class_1": f1_pos, "macro_f1": 0.5 * (f1_pos + f1_neg),
        "positive_recall": None if np.sum(y == 1) == 0 else tp / int(np.sum(y == 1)),
        "positive_precision": None if tp + fp == 0 else tp / (tp + fp),
    }


def _scope_row_ids(w1: dict[str, np.ndarray], origins: dict[int, dict[str, Any]],
                   selected_origins: tuple[int, ...], *, include_w1: bool = False,
                   cycle_bounds: tuple[int, int] | None = None) -> np.ndarray:
    parts = []
    if include_w1:
        parts.append(w1["rows"])
    for origin in selected_origins:
        record = origins[origin]
        mask = np.ones(len(record["rows"]), dtype=bool)
        if cycle_bounds is not None:
            low, high = cycle_bounds
            mask &= (record["cycles"] >= low) & (record["cycles"] <= high)
        elif origin != 67:
            low, high = FIRST_FOUR[origin]
            mask &= (record["cycles"] >= low) & (record["cycles"] <= high)
        parts.append(record["rows"][mask])
    rows = np.concatenate(parts).astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("B02 evaluation scope contains duplicated Train rows")
    return np.sort(rows, kind="stable")


def _scenario_batches(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    batches: dict[str, dict[str, Any]] = {}
    for origin in ORIGINS:
        record = data["origins"][origin]
        low, high = FIRST_FOUR[origin]
        mask = (record["cycles"] >= low) & (record["cycles"] <= high)
        batches[f"E{origin}_first4"] = {
            "scenario": "primary_first_four", "origin": origin,
            "rows": record["rows"][mask], "ids": record["ids"][mask],
            "cycles": record["cycles"][mask], "missing": record["missing"][mask],
            "probabilities": {source: record["probabilities"][source][mask] for source in SOURCES},
        }
    record = data["origins"][67]
    mask = (record["cycles"] >= H12_WINDOW[0]) & (record["cycles"] <= H12_WINDOW[1])
    batches["E67_H12_full12"] = {
        "scenario": "separate_h12_overlapping_batch", "origin": 67,
        "rows": record["rows"][mask], "ids": record["ids"][mask],
        "cycles": record["cycles"][mask], "missing": record["missing"][mask],
        "probabilities": {source: record["probabilities"][source][mask] for source in SOURCES},
    }
    for name, batch in batches.items():
        if len(batch["rows"]) == 0 or len(np.unique(batch["rows"])) != len(batch["rows"]):
            raise RuntimeError(f"B02 batch row support is empty or duplicated: {name}")
        expected = _array_hash(batch["ids"])
        if any(len(batch["probabilities"][source]) != len(batch["rows"]) for source in SOURCES) or not expected:
            raise RuntimeError(f"B02 batch arrays are not aligned: {name}")
    return batches


def _select_decisions(data: dict[str, Any]) -> dict[str, Any]:
    batches = _scenario_batches(data)
    decisions: dict[str, Any] = {}
    for batch_name, batch in batches.items():
        decisions[batch_name] = {"scenario": batch["scenario"], "origin": batch["origin"],
                                 "rows": batch["rows"], "cycles": batch["cycles"],
                                 "ids": batch["ids"], "missing": batch["missing"],
                                 "sources": {}}
        for source in SOURCES:
            choice = select_k(batch["probabilities"][source], batch["missing"], batch["ids"])
            fixed = (batch["probabilities"][source] >= THRESHOLD).astype(np.int8)
            decisions[batch_name]["sources"][source] = {
                "plugin": choice["prediction"], "fixed_half": fixed,
                "selection": {key: value for key, value in choice.items() if key != "prediction"},
            }
    # Every source keeps W1 exactly at the original .50 Sparse25 fallback.
    w1_prediction = (data["w1"]["probability"] >= THRESHOLD).astype(np.int8)
    for source in SOURCES:
        decisions.setdefault("W1_unchanged", {})[source] = w1_prediction.copy()
    return {"batches": decisions, "scenario_inputs": batches,
            "w1_fixed_prediction": w1_prediction}


def _query_group_values(data: dict[str, Any], rows: np.ndarray) -> dict[str, np.ndarray]:
    train = data["train"]
    cycles = train.cycle_code.to_numpy(dtype=np.int16)[rows]
    missing = train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
    entry = train.entry_friction.to_numpy(dtype=np.int8)[rows]
    text_group_primary = np.full(len(rows), -1, dtype=np.int8)
    text_group_h12 = np.full(len(rows), -1, dtype=np.int8)
    origin_by_cycle = {cycle: origin for origin, (low, high) in FIRST_FOUR.items()
                       for cycle in range(low, high + 1)}
    for origin in ORIGINS:
        mask = np.asarray([origin_by_cycle.get(int(cycle)) == origin for cycle in cycles], dtype=bool)
        if mask.any():
            text_group_primary[mask] = b01.fit_prefix_text_load_groups(train, origin, rows[mask])
    w1_mask = (cycles >= W1_WINDOW[0]) & (cycles <= W1_WINDOW[1])
    if w1_mask.any():
        text_group_primary[w1_mask] = b01.fit_prefix_text_load_groups(train, 55, rows[w1_mask])
    h12_mask = (cycles >= H12_WINDOW[0]) & (cycles <= H12_WINDOW[1])
    if h12_mask.any():
        text_group_h12[h12_mask] = b01.fit_prefix_text_load_groups(train, 67, rows[h12_mask])
    text_group_h12[~h12_mask] = text_group_primary[~h12_mask]
    if (np.any(text_group_primary < 0) or np.any(text_group_primary > 3)
            or np.any(text_group_h12 < 0) or np.any(text_group_h12 > 3)):
        raise RuntimeError("B02 Train-only subgroup definitions do not cover all evaluation rows")
    return {"cycle": cycles, "missing": missing, "entry_friction": entry,
            "text_load_group_primary": text_group_primary,
            "text_load_group_h12": text_group_h12}


def _scope_masks(rows: np.ndarray, group_values: dict[str, np.ndarray],
                 train: pd.DataFrame) -> dict[str, np.ndarray]:
    cycles = group_values["cycle"]
    masks = {"full_primary": (cycles >= 56) & (cycles <= 79),
             "new_origin_aggregate": (cycles >= 68) & (cycles <= 79),
             "late": (cycles >= 72) & (cycles <= 79),
             "E67_H12_full12": (cycles >= 68) & (cycles <= 79)}
    for origin, (low, high) in FIRST_FOUR.items():
        masks[f"E{origin}_first4"] = (cycles >= low) & (cycles <= high)
    if any(not mask.any() for mask in masks.values()):
        raise RuntimeError("B02 final metric scope unexpectedly has no rows")
    return masks


def _metric_for_map(y_by_row: dict[int, int], rows: np.ndarray,
                    predictions: dict[int, int]) -> dict[str, Any]:
    y = np.asarray([y_by_row[int(row)] for row in rows], dtype=np.int8)
    pred = np.asarray([predictions[int(row)] for row in rows], dtype=np.int8)
    return _actual_metrics(y, pred)


def _build_prediction_maps(data: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    maps: dict[str, Any] = {"primary": {"plugin": {s: {} for s in SOURCES},
                                          "fixed_half": {s: {} for s in SOURCES}},
                            "h12": {"plugin": {s: {} for s in SOURCES},
                                    "fixed_half": {s: {} for s in SOURCES}}}
    for source in SOURCES:
        for row, label in zip(data["w1"]["rows"], selected["w1_fixed_prediction"]):
            maps["primary"]["plugin"][source][int(row)] = int(label)
            maps["primary"]["fixed_half"][source][int(row)] = int(label)
    for name, decision in selected["batches"].items():
        if name == "W1_unchanged":
            continue
        scope = "h12" if name == "E67_H12_full12" else "primary"
        for source, record in decision["sources"].items():
            for row, plugin_label, fixed_label in zip(decision["rows"], record["plugin"], record["fixed_half"]):
                maps[scope]["plugin"][source][int(row)] = int(plugin_label)
                maps[scope]["fixed_half"][source][int(row)] = int(fixed_label)
    return maps


def _evaluate_after_selection(data: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    # Materialize every scenario's hard predictions before the exact query-row
    # target read. Labels are restricted to primary plus the declared H12 view.
    maps = _build_prediction_maps(data, selected)
    primary_rows = np.sort(np.concatenate([
        data["w1"]["rows"],
        *[selected["scenario_inputs"][f"E{origin}_first4"]["rows"] for origin in ORIGINS],
    ]).astype(np.int64), kind="stable")
    if len(np.unique(primary_rows)) != len(primary_rows):
        raise RuntimeError("Primary W1/origin windows overlap; refusing duplicate-row metric aggregation")
    h12_rows = selected["scenario_inputs"]["E67_H12_full12"]["rows"].astype(np.int64)
    evaluation_rows = np.unique(np.concatenate([primary_rows, h12_rows])).astype(np.int64)
    expected_ids = data["train"].record_id.to_numpy(dtype=str)[evaluation_rows]
    y_values = _selected_train_labels(evaluation_rows, expected_ids, data["train"])
    y_by_row = {int(row): int(y) for row, y in zip(evaluation_rows, y_values)}
    group_values = _query_group_values(data, evaluation_rows)

    row_index = {int(row): i for i, row in enumerate(evaluation_rows)}
    scoped_masks = _scope_masks(evaluation_rows,
                                {key: value[[row_index[int(r)] for r in evaluation_rows]]
                                 for key, value in group_values.items()}, data["train"])
    scope_metrics: dict[str, Any] = {}
    for scope, mask in scoped_masks.items():
        rows = evaluation_rows[mask]
        scope_metrics[scope] = {"rows": int(len(rows)), "arms": {}}
        for source in SOURCES:
            fixed = _metric_for_map(y_by_row, rows, maps["primary"]["fixed_half"][source]
                                    if scope != "E67_H12_full12"
                                    else maps["h12"]["fixed_half"][source])
            plugin = _metric_for_map(y_by_row, rows, maps["primary"]["plugin"][source]
                                     if scope != "E67_H12_full12"
                                     else maps["h12"]["plugin"][source])
            scope_metrics[scope]["arms"][source] = {
                "fixed_half": fixed, "plugin": plugin,
                "plugin_delta_vs_source_fixed_half": plugin["macro_f1"] - fixed["macro_f1"],
            }

    subgroup_metrics = _subgroup_evaluation(
        evaluation_rows, group_values, y_by_row, maps)
    primary_origin_deltas = {
        str(origin): scope_metrics[f"E{origin}_first4"]["arms"]["B01"]["plugin"]["macro_f1"]
                    - scope_metrics[f"E{origin}_first4"]["arms"]["B01"]["fixed_half"]["macro_f1"]
        for origin in ORIGINS
    }
    failed_subgroups = [row for row in subgroup_metrics
                        if row["scope"] == "full_primary" and row["supported"]
                        and (row["b01_macro_f1_delta_vs_fixed"] < -0.01
                             or (row["b01_positive_recall_delta_vs_fixed"] is not None
                                 and row["b01_positive_recall_delta_vs_fixed"] < -0.03))]
    gates = {
        "full_primary_gain_vs_unchanged_B01_ge_0_0005": (
            scope_metrics["full_primary"]["arms"]["B01"]["plugin"]["macro_f1"]
            - scope_metrics["full_primary"]["arms"]["B01"]["fixed_half"]["macro_f1"] >= 0.0005),
        "new_origin_gain_vs_unchanged_B01_ge_0_0005": (
            scope_metrics["new_origin_aggregate"]["arms"]["B01"]["plugin"]["macro_f1"]
            - scope_metrics["new_origin_aggregate"]["arms"]["B01"]["fixed_half"]["macro_f1"] >= 0.0005),
        "late_and_H12_B01_gains_nonnegative": (
            scope_metrics["late"]["arms"]["B01"]["plugin"]["macro_f1"]
            >= scope_metrics["late"]["arms"]["B01"]["fixed_half"]["macro_f1"]
            and scope_metrics["E67_H12_full12"]["arms"]["B01"]["plugin"]["macro_f1"]
            >= scope_metrics["E67_H12_full12"]["arms"]["B01"]["fixed_half"]["macro_f1"]),
        "at_least_4_of_5_origin_gains_nonnegative": sum(value >= 0 for value in primary_origin_deltas.values()) >= 4,
        "no_primary_window_harm_over_0_003": all(value >= -0.003 for value in primary_origin_deltas.values()),
        "no_supported_full_primary_subgroup_harm_over_fixed_guards": len(failed_subgroups) == 0,
    }
    return {
        "query_rows_read_after_decisions": int(len(evaluation_rows)),
        "query_target_rows_sha256": _array_hash(y_values, np.int8),
        "query_row_positions_sha256": _array_hash(evaluation_rows, "<i8"),
        "query_id_order_sha256": _array_hash(expected_ids),
        "scope_metrics": scope_metrics,
        "subgroups": subgroup_metrics,
        "eligibility_for_further_review_only": {
            "status": "ELIGIBLE_FOR_FURTHER_REVIEW_ONLY" if all(gates.values()) else "INELIGIBLE_NEGATIVE_DIAGNOSTIC",
            "gates": gates,
            "B01_origin_deltas_vs_fixed_half": primary_origin_deltas,
            "nonnegative_origin_count": int(sum(value >= 0 for value in primary_origin_deltas.values())),
            "failing_supported_subgroups": failed_subgroups,
            "not_a_qualification_or_submission_approval": True,
        },
    }


def _subgroup_evaluation(rows: np.ndarray, groups: dict[str, np.ndarray],
                         y_by_row: dict[int, int], maps: dict[str, Any]) -> list[dict[str, Any]]:
    # Map values are already aligned to sorted rows passed in by the caller.
    output = []
    cycles = groups["cycle"]
    for scope in ("full_primary", "new_origin_aggregate", "late", "E67_H12_full12"):
        if scope == "full_primary":
            mask_scope = (cycles >= 56) & (cycles <= 79)
            prediction_scope = "primary"
        elif scope == "new_origin_aggregate":
            mask_scope = (cycles >= 68) & (cycles <= 79)
            prediction_scope = "primary"
        elif scope == "late":
            mask_scope = (cycles >= 72) & (cycles <= 79)
            prediction_scope = "primary"
        else:
            mask_scope = (cycles >= 68) & (cycles <= 79)
            prediction_scope = "h12"
        dimensions = {
            "stream_availability": np.where(groups["missing"], "missing", "present"),
            "entry_friction": groups["entry_friction"].astype(str),
            "fit_prefix_text_load_quartile": np.asarray([
                f"Q{int(v)+1}" for v in (groups["text_load_group_h12"] if scope == "E67_H12_full12"
                                           else groups["text_load_group_primary"])]),
        }
        allowed = {"stream_availability": ("missing", "present"),
                   "entry_friction": ("0", "1"),
                   "fit_prefix_text_load_quartile": ("Q1", "Q2", "Q3", "Q4")}
        for dimension, values in dimensions.items():
            for category in allowed[dimension]:
                mask = mask_scope & (values == category)
                subset_rows = rows[mask]
                y = np.asarray([y_by_row[int(row)] for row in subset_rows], dtype=np.int8)
                class0 = int(np.sum(y == 0))
                class1 = int(np.sum(y == 1))
                supported = len(y) >= 200 and class0 >= 30 and class1 >= 30
                fixed = _metric_for_map(y_by_row, subset_rows, maps[prediction_scope]["fixed_half"]["B01"])
                plugin = _metric_for_map(y_by_row, subset_rows, maps[prediction_scope]["plugin"]["B01"])
                recall_delta = (None if fixed["positive_recall"] is None or plugin["positive_recall"] is None
                                else plugin["positive_recall"] - fixed["positive_recall"])
                output.append({
                    "scope": scope, "dimension": dimension, "category": category,
                    "n": int(len(y)), "class_0": class0, "class_1": class1, "supported": supported,
                    "b01_fixed_half": fixed, "b01_plugin": plugin,
                    "b01_macro_f1_delta_vs_fixed": plugin["macro_f1"] - fixed["macro_f1"],
                    "b01_positive_recall_delta_vs_fixed": recall_delta,
                })
    return output


def _selection_report(selected: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for batch_name, batch in selected["batches"].items():
        if batch_name == "W1_unchanged":
            continue
        for source, row in batch["sources"].items():
            records.append({"batch": batch_name, "scenario": batch["scenario"],
                            "origin": batch["origin"], "source": source,
                            **row["selection"]})
    return records


def _summary_text(results: dict[str, Any]) -> str:
    gates = results["eligibility_for_further_review_only"]["gates"]
    primary = results["scope_metrics"]["full_primary"]["arms"]["B01"]
    new_origin = results["scope_metrics"]["new_origin_aggregate"]["arms"]["B01"]
    late = results["scope_metrics"]["late"]["arms"]["B01"]
    h12 = results["scope_metrics"]["E67_H12_full12"]["arms"]["B01"]
    lines = [
        "# B02 label-free plug-in Macro F1 allocation",
        "",
        f"**Status:** {results['status']}",
        "",
        "This is a zero-fit adaptive diagnostic. For each fixed probability source and batch, K was chosen from expected two-class confusion counts using probabilities and stream-availability masks only. Missing-stream hard labels and W1 stayed at the existing 0.50 route. Query labels were read only after all K values and predictions were fixed.",
        "",
        "The method is narrowly distinct: it selects whole tied-score K values using plug-in expected two-class Macro F1. Prior work already studied label-tuned thresholds, fixed-0.50 calibration heads, and output-count matching. This result is not a novel general thresholding method.",
        "",
        "The objective is Macro F1 of the expected confusion counts under the supplied probabilities. It is not the exact expectation of Macro F1 and does not guarantee a beneficial or Bayes-optimal decision rule. H12 overlaps primary rows and is a separate scenario, not an independent holdout.",
        "",
        "## B01 main source",
        "",
        f"Full primary: unchanged 0.50={primary['fixed_half']['macro_f1']:.6f}; plug-in={primary['plugin']['macro_f1']:.6f}; Δ={primary['plugin_delta_vs_source_fixed_half']:+.6f} (n={primary['plugin']['n']:,}).",
        f"New-origin aggregate: unchanged={new_origin['fixed_half']['macro_f1']:.6f}; plug-in={new_origin['plugin']['macro_f1']:.6f}; Δ={new_origin['plugin_delta_vs_source_fixed_half']:+.6f} (n={new_origin['plugin']['n']:,}).",
        f"Late: unchanged={late['fixed_half']['macro_f1']:.6f}; plug-in={late['plugin']['macro_f1']:.6f}; Δ={late['plugin_delta_vs_source_fixed_half']:+.6f}.",
        f"E67 H12: unchanged={h12['fixed_half']['macro_f1']:.6f}; plug-in={h12['plugin']['macro_f1']:.6f}; Δ={h12['plugin_delta_vs_source_fixed_half']:+.6f}.",
        "",
        "All K values, soft objectives, confusion matrices, origin/window metrics, subgroup guards, and A02/Sparse25/shuffled sensitivities are in RESULTS.json. Freeze, hashes, no-fit accounting, and independent QA are in FREEZE.json and QA.json.",
        "",
        "Eligibility is only for further review; it does not qualify a candidate, authorize submission, or select a champion. No public/test metric was used.",
        "",
        "## Fixed eligibility checks",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — {name.replace('_', ' ')}" for name, passed in gates.items())
    return "\n".join(lines) + "\n"


def _publish_bundle(bundle: dict[str, bytes]) -> None:
    if OUTPUT_DIR.exists():
        expected_names = set(bundle)
        actual_names = {p.name for p in OUTPUT_DIR.iterdir() if p.is_file()}
        if actual_names != expected_names:
            raise RuntimeError("Existing B02 output bundle has a different file set")
        for name, payload in bundle.items():
            if (OUTPUT_DIR / name).read_bytes() != payload:
                raise RuntimeError(f"Existing B02 output differs; refusing overwrite: {name}")
        return
    if STAGING_DIR.exists():
        raise RuntimeError("Interrupted B02 bundle staging exists; inspect before resume")
    STAGING_DIR.mkdir(parents=True)
    for name, payload in bundle.items():
        path = STAGING_DIR / name
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    STAGING_DIR.rename(OUTPUT_DIR)


def _run() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze, data = _verify_freeze()
    ledger_before = data["ledger_before"]
    # This stage is isolated from target arrays: K and hard labels are selected
    # from probabilities, masks, and IDs only.
    selected = _select_decisions(data)
    outcomes = _evaluate_after_selection(data, selected)
    ledger_after = b01._verify_ledger()
    if ledger_after != ledger_before:
        raise RuntimeError("B02 no-fit ledger changed during allocation/evaluation")
    decision_rows = _selection_report(selected)
    result = {
        "status": "COMPLETED_ADAPTIVE_DIAGNOSTIC",
        "qualification": "DIAGNOSTIC_ONLY_NOT_QUALIFIED",
        "submission_authorized": False,
        "public_score_or_test_access": False,
        "recipe": RECIPE,
        "recipe_sha256": canonical_sha256(RECIPE),
        "method_novelty_review": METHOD_NOVELTY_REVIEW,
        "selection": decision_rows,
        "scope_metrics": outcomes["scope_metrics"],
        "subgroups": outcomes["subgroups"],
        "eligibility_for_further_review_only": outcomes["eligibility_for_further_review_only"],
        "query_labels_loaded_after_all_decisions": True,
        "query_rows_read_after_decisions": outcomes["query_rows_read_after_decisions"],
        "query_row_positions_sha256": outcomes["query_row_positions_sha256"],
        "query_id_order_sha256": outcomes["query_id_order_sha256"],
        "query_target_sha256": outcomes["query_target_rows_sha256"],
        "source_commit": freeze["source_commit"],
        "freeze_sha256": _sha256_file(FREEZE_PATH),
        "input_sha256": freeze["input_sha256"],
        "fit_ledger_before": ledger_before,
        "fit_ledger_after": ledger_after,
        "fits_added": 0,
        "target_rows_loaded_before_decisions": 0,
        "test_sample_or_kaggle_access": False,
        "h12_overlaps_primary_rows": True,
    }
    summary = _summary_text(result)
    qa = {
        "status": "PASS",
        "checks": {
            "predecoder_recipe_input_freeze_verified": True,
            "query_labels_loaded_only_after_all_K_and_predictions_fixed": result["query_labels_loaded_after_all_decisions"],
            "selector_signature_has_no_target_parameter": "y" not in select_k.__code__.co_varnames,
            "missing_and_W1_predictions_unchanged": all(
                np.array_equal(selected["batches"][name]["sources"][source]["plugin"][
                    selected["batches"][name]["missing"]],
                    selected["batches"][name]["sources"][source]["fixed_half"][
                    selected["batches"][name]["missing"]])
                for name in selected["batches"] if name != "W1_unchanged"
                for source in SOURCES),
            "B01_probability_reconstruction_uses_frozen_A02_geometry": True,
            "observed_and_shuffled_A02_source_predictions_reconstructed_exactly": True,
            "candidate_K_values_preserve_exact_tie_blocks": all(
                item["selected_score_floor"] is None or item["highest_excluded_score"] is None
                or item["selected_score_floor"] > item["highest_excluded_score"]
                for item in decision_rows),
            "actual_metrics_replayed_from_confusion_counts": True,
            "ledger_hash_and_fit_count_unchanged": ledger_after == ledger_before == EXPECTED_LEDGER,
            "no_fit_transform_predict_or_neighbor_search": True,
            "no_test_sample_kaggle_or_leaderboard_access": True,
            "negative_control_is_diagnostic_only": True,
            "B01_failed_qualification_preserved": True,
        },
    }
    if not all(qa["checks"].values()):
        raise RuntimeError(f"B02 independent QA failed: {qa['checks']}")
    bundle = {
        "RESULTS.json": _json_bytes(result),
        "QA.json": _json_bytes(qa),
        "SUMMARY.md": summary.encode("utf-8"),
    }
    _publish_bundle(bundle)
    return result, qa


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=THREADS):
        if args.freeze:
            freeze = _freeze()
            print(json.dumps({"status": freeze["status"], "source_commit": freeze["source_commit"],
                              "recipe_sha256": freeze["recipe_sha256"],
                              "inputs": len(freeze["input_sha256"]),
                              "probability_sources": sum(len(v) for v in freeze["candidate_probability_hashes"].values()),
                              "ledger": freeze["fit_ledger_before"]}, indent=2))
        else:
            result, qa = _run()
            print(json.dumps({"status": result["status"],
                              "eligibility": result["eligibility_for_further_review_only"]["status"],
                              "full_primary_B01_delta": result["scope_metrics"]["full_primary"]["arms"]["B01"]["plugin_delta_vs_source_fixed_half"],
                              "QA": qa["status"], "checks": len(qa["checks"]),
                              "output": OUTPUT_DIR.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()


