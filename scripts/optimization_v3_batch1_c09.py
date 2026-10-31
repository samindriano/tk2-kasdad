"""C09 fit, frozen-route evaluation, and read-only preflight.

Default operation is read-only preflight. Official fitting/scoring requires an
explicit --run after a separately written freeze; this script never accesses
official Test, SampleSubmission, or Kaggle.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from src.models.classification_metrics import binary_metrics
from src.models.optimization_v3_batch1_c09 import (
    C09TextEntryFrictionClassifier,
    entry_friction_one_mask,
)
from src.models.classification_models import _LR_PARAMS, _TFIDF_PARAMS, _text_values
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import _number_array
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    exclusive_training_slot,
    reserve_card_budget,
)
from src.utils.paths import task_files


CARD = ROOT / "reports/optimization_v3/research_backlog/cards/C09.md"
QUEUE = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
BASELINE_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
PILOT_CONFIG = ROOT / "experiments/optimization_v3/classification_sparse"
PILOT_REPORT = ROOT / "reports/optimization_v3/classification_sparse"
PILOT_CACHE = ROOT / "experiments/runs/optimization_v3_classification_sparse"
PILOT_FULL_CONFIG = ROOT / "experiments/optimization_v3/classification_sparse_full"
PILOT_FULL_REPORT = ROOT / "reports/optimization_v3/classification_sparse_full"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CB_CACHE = ROOT / "experiments/runs/dual_cls_catboost_v1"
CONFIG = ROOT / "experiments/optimization_v3/batch1/C09"
REPORT = ROOT / "reports/optimization_v3/batch1/C09"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C09"
TRAINING_ROOT = RUN_DIR.parent
TRAINING_LOCK_PATH = TRAINING_ROOT / ".training.lock"
BATCH_LEDGER_PATH = TRAINING_ROOT / "batch_ledger.jsonl"
RECIPE_PATH = CONFIG / "recipe.json"
FREEZE_PATH = CONFIG / "freeze.json"
RESULTS_PATH = REPORT / "results.json"
VERIFICATION_PATH = REPORT / "verification.json"
SUMMARY_PATH = REPORT / "summary.md"

EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
FIT_ENDS = (55, 67)
HORIZON_BY_END = {55: 67, 67: 79}
PRIMARY_BY_END = {55: (56, 59, "W1"), 67: (68, 71, "W4")}
THRESHOLD = 0.50
PRESENT_C09_WEIGHT = 0.75
PRESENT_SPARSE_WEIGHT = 0.25
MAX_JOB_RSS_GIB = 6.0
MIN_FREE_RAM_GIB = 4.0
MEMORY_HEADROOM_GIB = 2.0
EXPECTED_CHANGE = (
    "Use original C2_B hybrid design and LR parameters; append only "
    "TF-IDF*I(entry_friction==1), retaining main TF-IDF and structured columns. "
    "Reference coding avoids duplicating both level blocks."
)
EXPECTED_PREPROCESS_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")

SOURCE_FILES = (
    "src/models/optimization_v3_batch1_c09.py",
    "scripts/optimization_v3_batch1_c09.py",
    "tests/test_optimization_v3_batch1_c09.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/models/optimization_v3_batch1.py",
    "src/utils/paths.py",
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_verify_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
    "scripts/research_campaign_classification_stage2.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive_atomic(path, payload)


def _save_json_atomic_replace(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace an owned progress receipt, never a source artifact."""
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _save_or_verify_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise ValueError(f"Existing C09 owned artifact differs; refusing overwrite: {path}")
        return
    _save_json_exclusive(path, value)


def _write_exclusive_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link creation is atomic and fails if target exists.
        os.link(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _save_npz_exclusive(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        with temp_path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link creation is atomic and fails if target exists.
        os.link(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _canonical_card() -> tuple[dict[str, Any], dict[str, Any]]:
    queue = _read_json(QUEUE)
    cards = [card for card in queue.get("cards", []) if card.get("id") == "C09"]
    if len(cards) != 1:
        raise ValueError("Canonical queue must contain exactly one C09 card")
    card = cards[0]
    expected = {
        "task": "classification",
        "change": EXPECTED_CHANGE,
        "policy": "C_LR_REPLACE",
        "pilot_predictor_fits": 2,
        "evaluation": "STANDARD",
        "condition": "No additional dependency",
        "status": "PLANNED_NOT_RUN",
        "implementation": "NOT_IMPLEMENTED",
    }
    for key, value in expected.items():
        if card.get(key) != value:
            raise ValueError(f"Canonical C09 {key} differs from the frozen recipe")
    if card.get("preflight_checks") != [
        "Use entry_friction, not telemetry_flag; convergence and sparse memory checks"
    ]:
        raise ValueError("C09 preflight checks differ from the canonical queue")
    if queue.get("baseline", {}).get("classification", {}).get("Train_sha256") != EXPECTED_TRAIN_SHA256:
        raise ValueError("Canonical classification Train hash differs")
    if queue.get("baseline", {}).get("classification", {}).get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Canonical C09 baseline manifest hash differs")
    if queue.get("policies", {}).get("C_LR_REPLACE") != (
        ".75 new hybrid LR +.25 frozen sparse binary on present; frozen CB missing; threshold.50"
    ):
        raise ValueError("Canonical C_LR_REPLACE policy differs")
    if queue.get("evaluation", {}).get("STANDARD", {}).get("pilot_cutoffs") != [55, 67]:
        raise ValueError("Canonical C09 pilot cutoffs differ")
    return queue, card


def _load_train() -> tuple[Path, pd.DataFrame, np.ndarray]:
    paths = task_files("classification")
    train_path = paths.get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Classification Train.csv was not found")
    train_path = train_path.resolve()
    train_hash = _sha256(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Official Classification Train hash differs: {train_hash}")
    frame = pd.read_csv(train_path)
    required = {
        "record_id", "cycle_code", "artifact_signal", "entry_friction", "signal_stream",
        "text_load", "telemetry_flag",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Official Train is missing required columns: {missing}")
    if len(frame) != 50_608:
        raise ValueError(f"Official classification Train row count differs: {len(frame)}")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    if cycles.min() != 47 or cycles.max() != 79 or not np.all(np.isin(np.unique(cycles), np.arange(47, 80))):
        raise ValueError("Official classification Train cycle range/continuity differs")
    if frame["record_id"].duplicated().any():
        raise ValueError("Official Train record_id must be unique")
    y_float = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y_float).all() or not np.isin(y_float, (0.0, 1.0)).all():
        raise ValueError("Official Train artifact_signal is not finite binary data")
    return train_path, frame, y_float.astype(np.int8)


def _expected_rows(frame: pd.DataFrame, fit_end: int, horizon: int) -> np.ndarray:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    return np.flatnonzero((cycles > fit_end) & (cycles <= horizon))


def _load_c2b(frame: pd.DataFrame, fit_end: int, horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = ROUND1_CACHE / f"fit{fit_end}__C2_B.npz"
    audit_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    audit_root = _read_json(audit_path)
    audit = audit_root.get("C2_B")
    if not isinstance(audit, dict) or audit.get("model") != "C2_B":
        raise ValueError(f"C2_B audit identity mismatch at {fit_end}")
    with np.load(cache, allow_pickle=False) as archive:
        rows = np.asarray(archive["outer_rows"], dtype=np.int64).copy()
        probability = np.asarray(archive["probability"], dtype=np.float64).copy()
    expected = _expected_rows(frame, fit_end, horizon)
    if not np.array_equal(rows, expected) or probability.shape != rows.shape:
        raise ValueError(f"C2_B validation rows differ at cutoff {fit_end}")
    if int(audit.get("fit_rows", -1)) != int(pd.to_numeric(frame.cycle_code).between(47, fit_end).sum()):
        raise ValueError(f"C2_B fit row count differs at cutoff {fit_end}")
    if audit.get("fit_cycles") != [47, fit_end] or audit.get("validation_prediction_cycles") != [fit_end + 1, horizon]:
        raise ValueError(f"C2_B temporal audit differs at cutoff {fit_end}")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"C2_B probabilities are invalid at cutoff {fit_end}")
    if array_digest(probability) != audit.get("probability_sha256"):
        raise ValueError(f"C2_B probability hash mismatch at cutoff {fit_end}")
    fitted_state = audit.get("fitted_state")
    if not isinstance(fitted_state, dict):
        raise ValueError(f"C2_B fitted-state details are missing at cutoff {fit_end}")
    transformed = fitted_state.get("transformed_features")
    text_state = fitted_state.get("text")
    if not isinstance(transformed, dict) or not isinstance(text_state, dict):
        raise ValueError(f"C2_B sparse feature details are missing at cutoff {fit_end}")
    encoded_counts = transformed.get("encoded_category_counts")
    numeric_count = transformed.get("numeric_count")
    vocabulary_size = text_state.get("vocabulary_size")
    if (not isinstance(encoded_counts, list) or not all(isinstance(value, int) for value in encoded_counts)
            or not isinstance(numeric_count, int) or not isinstance(vocabulary_size, int)):
        raise ValueError(f"C2_B sparse feature dimensions are malformed at cutoff {fit_end}")
    return rows, probability, {
        "cache_sha256": _sha256(cache),
        "audit_sha256": _sha256(audit_path),
        "probability_sha256": array_digest(probability),
        "fit_rows": int(audit["fit_rows"]),
        "structured_numeric_width": numeric_count,
        "structured_ohe_width_by_column": [int(value) for value in encoded_counts],
        "text_vocabulary_size": vocabulary_size,
    }


def _load_cb(rows: np.ndarray, fit_end: int) -> tuple[np.ndarray, dict[str, str]]:
    cache = CB_CACHE / f"fit{fit_end}.npz"
    sidecar = cache.with_suffix(".json")
    metadata = _read_json(sidecar)
    if metadata.get("npz_sha256") != _sha256(cache):
        raise ValueError(f"CatBoost cache checksum mismatch at cutoff {fit_end}")
    with np.load(cache, allow_pickle=False) as archive:
        cb_rows = np.asarray(archive["rows"], dtype=np.int64).copy()
        probability = np.asarray(archive["candidate"], dtype=np.float64).copy()
    if not np.array_equal(cb_rows, rows) or probability.shape != rows.shape:
        raise ValueError(f"CatBoost rows do not align at cutoff {fit_end}")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"CatBoost probabilities are invalid at cutoff {fit_end}")
    if array_digest(probability) != metadata.get("prediction_sha256"):
        raise ValueError(f"CatBoost prediction hash mismatch at cutoff {fit_end}")
    return probability, {"npz_sha256": _sha256(cache), "sidecar_sha256": _sha256(sidecar)}


def _load_sparse_pilot(
    frame: pd.DataFrame,
    rows: np.ndarray,
    fit_end: int,
    pilot_freeze: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    cache = PILOT_CACHE / f"fit{fit_end}.npz"
    sidecar = cache.with_suffix(".json")
    metadata = _read_json(sidecar)
    if metadata.get("freeze_signature") != pilot_freeze.get("signature"):
        raise ValueError(f"Sparse pilot freeze signature differs at cutoff {fit_end}")
    if metadata.get("npz_sha256") != _sha256(cache):
        raise ValueError(f"Sparse pilot cache checksum mismatch at cutoff {fit_end}")
    with np.load(cache, allow_pickle=False) as archive:
        sparse_rows = np.asarray(archive["rows"], dtype=np.int64).copy()
        raw_probability = np.asarray(archive["raw_candidate"], dtype=np.float64).copy()
        blend25 = np.asarray(archive["blend25"], dtype=np.float64).copy()
    if not np.array_equal(sparse_rows, rows):
        raise ValueError(f"Sparse pilot rows do not align at cutoff {fit_end}")
    if raw_probability.shape != rows.shape or blend25.shape != rows.shape:
        raise ValueError(f"Sparse pilot vector shape differs at cutoff {fit_end}")
    for name, probability in (("raw sparse", raw_probability), ("blend25", blend25)):
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError(f"{name} probabilities are invalid at cutoff {fit_end}")
    return raw_probability, blend25, {"npz_sha256": _sha256(cache), "sidecar_sha256": _sha256(sidecar)}


def _baseline_files_identity() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Verify approved current-control artifacts and return hash-bound controls."""
    queue, card = _canonical_card()
    if _sha256(CARD) != _sha256(ROOT / card["card_path"]):
        raise AssertionError("C09 card path identity mismatch")
    if _sha256(BASELINE_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Accepted C2_B/sparse baseline manifest hash differs")
    manifest = _read_json(BASELINE_MANIFEST)
    if manifest.get("status") != "PREPARED_AND_QA_PASS" or manifest.get("candidate_id") != "v3_cls_sparse25":
        raise ValueError("Accepted classification baseline manifest status/identity differs")
    recipe = manifest.get("recipe", {})
    if recipe != {
        "kind": "availability_present_blend",
        "present": [{"model": "C2_B", "weight": 0.75}, {"model": "SPARSE_BINARY", "weight": 0.25}],
        "missing": "CB_MISSING",
        "threshold": 0.5,
    }:
        raise ValueError("Accepted baseline manifest route differs from C09 control")

    pilot_freeze = _read_json(PILOT_CONFIG / "freeze.json")
    pilot_verification = _read_json(PILOT_REPORT / "verification.json")
    if pilot_verification.get("status") != "PASS" or pilot_verification.get("freeze_signature") != pilot_freeze.get("signature"):
        raise ValueError("Frozen sparse pilot controls are not independently verified")
    full_freeze = _read_json(PILOT_FULL_CONFIG / "freeze.json")
    full_verification = _read_json(PILOT_FULL_REPORT / "verification.json")
    if full_verification.get("status") != "PASS" or full_verification.get("freeze_signature") != full_freeze.get("signature"):
        raise ValueError("Frozen sparse full-six controls are not independently verified")
    return queue, {
        "manifest": manifest,
        "pilot_freeze": pilot_freeze,
        "pilot_verification": pilot_verification,
        "full_freeze": full_freeze,
        "full_verification": full_verification,
    }


def _fit_quantile_labels(
    fit_values: np.ndarray,
    eval_values: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    fit = np.asarray(fit_values, dtype=float)
    eval_ = np.asarray(eval_values, dtype=float)
    fit = fit[np.isfinite(fit)]
    labels = np.full(len(eval_), "missing", dtype=object)
    valid = np.isfinite(eval_)
    if not len(fit):
        return labels, []
    edges = np.unique(np.quantile(fit, [0.0, 0.25, 0.50, 0.75, 1.0]))
    if len(edges) < 2:
        labels[valid] = "Q1"
    else:
        bins = np.searchsorted(edges[1:-1], eval_[valid], side="left")
        labels[valid] = np.asarray([f"Q{int(i) + 1}" for i in bins], dtype=object)
    return labels, [float(value) for value in edges]


def _group_labels(frame: pd.DataFrame, fit_rows: np.ndarray, rows: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    evaluation = frame.iloc[rows]
    ef1, _ = entry_friction_one_mask(evaluation)
    ef_values, ef_missing, ef_invalid = _number_array(evaluation["entry_friction"])
    ef_known = (ef_missing == 0) & (ef_invalid == 0) & np.isfinite(ef_values)
    ef0 = ef_known & (ef_values == 0.0)
    ef1_bool = ef1 & ef_known
    ef_unknown = ~(ef0 | ef1_bool)
    missing = stream_missing_mask(evaluation)
    fit_text, _, _ = _number_array(frame.iloc[fit_rows]["text_load"])
    eval_text, _, _ = _number_array(evaluation["text_load"])
    text_labels, edges = _fit_quantile_labels(fit_text, eval_text)
    labels = {
        "entry_friction": np.where(ef0, "EF0", np.where(ef1_bool, "EF1", "EF_UNKNOWN")),
        "stream": np.where(missing, "missing", "present"),
        "text_load": text_labels,
    }
    return labels, {"text_load": edges}


def _group_metric_table(
    y: np.ndarray,
    rows: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    gate = _canonical_card()[0]["evaluation"]["classification_gate"]
    minimum_n = int(gate["supported_group_n_min"])
    minimum_per_class = int(gate["positive_and_negative_per_group_min"])
    output: dict[str, Any] = {}
    actual = y[rows]
    for axis, group_labels in labels.items():
        if group_labels.shape != rows.shape:
            raise ValueError(f"Group labels do not align for {axis}")
        output[axis] = {}
        for group in sorted(set(str(value) for value in group_labels)):
            mask = np.asarray([str(value) == group for value in group_labels], dtype=bool)
            yy = actual[mask]
            p0 = baseline[mask]
            p1 = candidate[mask]
            negatives = int(np.count_nonzero(yy == 0))
            positives = int(np.count_nonzero(yy == 1))
            supported = int(mask.sum()) >= minimum_n and min(negatives, positives) >= minimum_per_class
            result: dict[str, Any] = {
                "n": int(mask.sum()),
                "negative_support": negatives,
                "positive_support": positives,
                "supported": bool(supported),
            }
            if mask.any():
                base_metrics = binary_metrics(yy, p0, threshold=THRESHOLD)
                candidate_metrics = binary_metrics(yy, p1, threshold=THRESHOLD)
                base_recall = float(base_metrics["per_class"][1]["recall"])
                candidate_recall = float(candidate_metrics["per_class"][1]["recall"])
                result.update({
                    "baseline_macro_f1": float(base_metrics["macro_f1"]),
                    "candidate_macro_f1": float(candidate_metrics["macro_f1"]),
                    "delta_macro_f1": float(candidate_metrics["macro_f1"] - base_metrics["macro_f1"]),
                    "macro_f1_harm": float(base_metrics["macro_f1"] - candidate_metrics["macro_f1"]),
                    "baseline_positive_recall": base_recall,
                    "candidate_positive_recall": candidate_recall,
                    "positive_recall_harm": float(base_recall - candidate_recall),
                })
                if supported:
                    result["gate_pass"] = bool(
                        result["macro_f1_harm"] <= float(gate["supported_group_F1_harm_max"])
                        and result["positive_recall_harm"] <= float(gate["positive_class_recall_harm_max"])
                    )
                else:
                    result["gate_pass"] = None
            else:
                result.update({"baseline_macro_f1": None, "candidate_macro_f1": None, "gate_pass": None})
            output[axis][group] = result
    return output


def _bundle(
    rows: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: dict[str, np.ndarray],
    edges: dict[str, list[float]],
    cycles: np.ndarray,
) -> dict[str, Any]:
    if not (rows.shape == baseline.shape == candidate.shape == cycles.shape):
        raise ValueError("C09 evaluation vectors must be row aligned")
    return {"rows": rows.copy(), "baseline": baseline.copy(), "candidate": candidate.copy(),
            "labels": {name: values.copy() for name, values in labels.items()},
            "edges": edges, "cycles": cycles.copy()}


def _subset_bundle(bundle: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != bundle["rows"].shape:
        raise ValueError("C09 view mask does not align to row bundle")
    return _bundle(bundle["rows"][keep], bundle["baseline"][keep], bundle["candidate"][keep],
                   {name: labels[keep] for name, labels in bundle["labels"].items()}, bundle["edges"],
                   bundle["cycles"][keep])


def _concat_bundles(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("Cannot aggregate an empty C09 view")
    rows = np.concatenate([part["rows"] for part in parts])
    if len(np.unique(rows)) != len(rows):
        raise ValueError("C09 pooled view has duplicate source rows")
    names = set(parts[0]["labels"])
    if any(set(part["labels"]) != names for part in parts):
        raise ValueError("C09 pooled subgroup schemas differ")
    labels = {name: np.concatenate([part["labels"][name] for part in parts]) for name in names}
    edges = {name: [part["edges"][name] for part in parts] for name in parts[0]["edges"]}
    return _bundle(rows, np.concatenate([p["baseline"] for p in parts]),
                   np.concatenate([p["candidate"] for p in parts]), labels, edges,
                   np.concatenate([p["cycles"] for p in parts]))


def _evaluate_bundle(y: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    rows = bundle["rows"]
    baseline = bundle["baseline"]
    candidate = bundle["candidate"]
    base_metrics = binary_metrics(y[rows], baseline, threshold=THRESHOLD)
    candidate_metrics = binary_metrics(y[rows], candidate, threshold=THRESHOLD)
    cycle_values = np.asarray(bundle["cycles"], dtype=int)
    per_cycle = []
    for cycle in sorted(np.unique(cycle_values)):
        mask = cycle_values == cycle
        per_cycle.append({
            "cycle": int(cycle),
            "n": int(mask.sum()),
            "baseline_macro_f1": float(binary_metrics(y[rows][mask], baseline[mask], THRESHOLD)["macro_f1"]),
            "candidate_macro_f1": float(binary_metrics(y[rows][mask], candidate[mask], THRESHOLD)["macro_f1"]),
        })
    group_metrics = _group_metric_table(y, rows, baseline, candidate, bundle["labels"])
    group_fails = [f"{axis}:{name}" for axis, groups in group_metrics.items() for name, record in groups.items()
                   if record.get("supported") and record.get("gate_pass") is False]
    return {
        "n": int(len(rows)),
        "threshold": THRESHOLD,
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "delta_macro_f1": float(candidate_metrics["macro_f1"] - base_metrics["macro_f1"]),
        "group_metrics": group_metrics,
        "group_gate_failures": group_fails,
        "per_cycle": per_cycle,
        "fit_defined_text_load_edges": bundle["edges"],
        "source_rows_sha256": array_digest(rows.astype(np.float64)),
    }


def _support_only(y: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    rows = bundle["rows"]
    actual = y[rows]
    result: dict[str, Any] = {"n": int(len(rows)), "fit_defined_text_load_edges": bundle["edges"], "groups": {}}
    for axis, labels in bundle["labels"].items():
        result["groups"][axis] = {}
        for group in sorted(set(str(value) for value in labels)):
            mask = np.asarray([str(value) == group for value in labels], dtype=bool)
            result["groups"][axis][group] = {
                "n": int(mask.sum()),
                "negative_support": int(np.count_nonzero(actual[mask] == 0)),
                "positive_support": int(np.count_nonzero(actual[mask] == 1)),
            }
    return result


def _load_control_fold(
    frame: pd.DataFrame,
    y: np.ndarray,
    fit_end: int,
    control_refs: dict[str, Any],
) -> dict[str, Any]:
    horizon = HORIZON_BY_END[fit_end]
    rows = _expected_rows(frame, fit_end, horizon)
    c2b_rows, c2b, c2b_identity = _load_c2b(frame, fit_end, horizon)
    if not np.array_equal(c2b_rows, rows):
        raise ValueError(f"C2_B rows differ from official future rows at {fit_end}")
    missing = stream_missing_mask(frame.iloc[rows])
    cb, cb_identity = _load_cb(rows, fit_end)
    sparse_raw, sparse_blend, sparse_identity = _load_sparse_pilot(
        frame, rows, fit_end, control_refs["pilot_freeze"]
    )
    expected_baseline = np.where(missing, cb, PRESENT_C09_WEIGHT * c2b + PRESENT_SPARSE_WEIGHT * sparse_raw)
    if not np.array_equal(expected_baseline, sparse_blend):
        raise ValueError(f"Cached accepted sparse25 route does not reconstruct at cutoff {fit_end}")
    if not np.array_equal(sparse_blend[missing], cb[missing]):
        raise ValueError(f"Accepted baseline missing-stream branch changed at cutoff {fit_end}")
    fit_rows = np.flatnonzero(pd.to_numeric(frame["cycle_code"], errors="raise").between(47, fit_end).to_numpy())
    labels, edges = _group_labels(frame, fit_rows, rows)
    return {
        "fit_end": fit_end,
        "horizon": horizon,
        "fit_rows": fit_rows,
        "rows": rows,
        "c2b": c2b,
        "sparse_raw": sparse_raw,
        "cb": cb,
        "missing": missing,
        "baseline": sparse_blend,
        "candidate": None,
        "labels": labels,
        "edges": edges,
        "identities": {"c2b": c2b_identity, "sparse": sparse_identity, "catboost": cb_identity},
    }


def _input_identity(train_path: Path, controls: dict[int, dict[str, Any]], control_refs: dict[str, Any]) -> dict[str, Any]:
    queue, card = _canonical_card()
    cache_paths = [CARD, QUEUE, BASELINE_MANIFEST,
                   PILOT_CONFIG / "freeze.json", PILOT_REPORT / "verification.json",
                   PILOT_FULL_CONFIG / "freeze.json", PILOT_FULL_REPORT / "verification.json"]
    for end in FIT_ENDS:
        cache_paths.extend([
            ROUND1_CACHE / f"fit{end}__C2_B.npz",
            ROUND1_AUDITS / f"fit{end}.json",
            PILOT_CACHE / f"fit{end}.npz",
            PILOT_CACHE / f"fit{end}.json",
            CB_CACHE / f"fit{end}.npz",
            CB_CACHE / f"fit{end}.json",
        ])
    missing = [str(p) for p in cache_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"C09 frozen inputs are missing: {missing}")
    source_missing = [p for p in SOURCE_FILES if not (ROOT / p).is_file()]
    if source_missing:
        raise FileNotFoundError(f"C09 source inputs are missing: {source_missing}")
    return {
        "recipe": {
            "card_id": "C09",
            "task": "classification",
            "policy": "C_LR_REPLACE",
            "change": EXPECTED_CHANGE,
            "pilot_fit_ends": list(FIT_ENDS),
            "horizons": {str(key): value for key, value in HORIZON_BY_END.items()},
            "primary": {str(key): list(value[:2]) for key, value in PRIMARY_BY_END.items()},
            "threshold": THRESHOLD,
            "present_weights": {"C09": PRESENT_C09_WEIGHT, "sparse_binary": PRESENT_SPARSE_WEIGHT},
            "missing_model": "frozen CatBoost cache",
        },
        "source_sha256": {path: _sha256(ROOT / path) for path in SOURCE_FILES},
        "inputs_sha256": {
            "official_train": _sha256(train_path),
            **{str(path.resolve().relative_to(ROOT)).replace("\\", "/"): _sha256(path)
               for path in cache_paths},
        },
        "baseline_manifest_sha256": control_refs["manifest_sha256"],
        "pilot_sparse_freeze_signature": control_refs["pilot_freeze"]["signature"],
        "pilot_sparse_verification_status": control_refs["pilot_verification"]["status"],
        "full_sparse_freeze_signature": control_refs["full_freeze"]["signature"],
        "full_sparse_verification_status": control_refs["full_verification"]["status"],
        "control_cache_checks": {str(end): controls[end]["identities"] for end in FIT_ENDS},
        "packages": {name: importlib.metadata.version(name) for name in EXPECTED_PREPROCESS_PACKAGES},
        "training_source_boundary": "official classification Train only; Test/Sample/Kaggle not read",
    }


def _available_ram_bytes() -> int | None:
    """Read current physical-memory availability without adding a dependency."""
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _sparse_memory_forecast(
    frame: pd.DataFrame,
    controls: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Conservative CSR design estimate; does not fit vocabulary or estimator."""
    analyzer = TfidfVectorizer(**_TFIDF_PARAMS).build_analyzer()
    vocabulary_cap = int(_TFIDF_PARAMS["max_features"])
    folds: dict[str, Any] = {}
    largest_peak = 0
    for end in FIT_ENDS:
        record = controls[end]
        fit_rows = record["fit_rows"]
        documents, _ = _text_values(frame.iloc[fit_rows]["signal_stream"])
        all_text_nnz = 0
        ef1_text_nnz = 0
        ef1, _ = entry_friction_one_mask(frame.iloc[fit_rows])
        for position, document in enumerate(documents):
            count = min(len(set(analyzer(document))), vocabulary_cap)
            all_text_nnz += count
            if ef1[position]:
                ef1_text_nnz += count
        rows = int(len(fit_rows))
        c2b_identity = record["identities"]["c2b"]
        structured_width_upper = int(c2b_identity["structured_numeric_width"])
        structured_width_upper += len(c2b_identity["structured_ohe_width_by_column"])
        structured_nnz_upper = rows * structured_width_upper
        missing_nnz_upper = rows
        base_nnz_upper = structured_nnz_upper + all_text_nnz + missing_nnz_upper
        interaction_nnz_upper = ef1_text_nnz
        extended_nnz_upper = base_nnz_upper + interaction_nnz_upper

        # Use 8-byte values, 8-byte indices and 8-byte row pointers. Count
        # simultaneous base blocks, sliced text, interaction, and hstack output.
        def csr_bytes(nnz: int, matrix_rows: int) -> int:
            return nnz * 16 + (matrix_rows + 1) * 8

        base_bytes = csr_bytes(base_nnz_upper, rows)
        structured_bytes = csr_bytes(structured_nnz_upper, rows)
        text_bytes = csr_bytes(all_text_nnz, rows)
        missing_bytes = csr_bytes(missing_nnz_upper, rows)
        interaction_bytes = csr_bytes(interaction_nnz_upper, rows)
        extended_bytes = csr_bytes(extended_nnz_upper, rows)
        peak = base_bytes + structured_bytes + 2 * text_bytes + missing_bytes + interaction_bytes + extended_bytes
        largest_peak = max(largest_peak, peak)
        folds[str(end)] = {
            "fit_rows": rows,
            "text_vocabulary_cap": vocabulary_cap,
            "analyzer_distinct_ngram_nnz_upper": all_text_nnz,
            "ef1_interaction_nnz_upper": ef1_text_nnz,
            "structured_width_upper": structured_width_upper,
            "structured_nnz_upper": structured_nnz_upper,
            "extended_nnz_upper": extended_nnz_upper,
            "conservative_simultaneous_csr_bytes": peak,
            "conservative_simultaneous_csr_gib": peak / (1024 ** 3),
            "estimator_internal_working_memory": "UNKNOWN; liblinear solver internals are outside CSR estimate",
        }
    available = _available_ram_bytes()
    available_gib = None if available is None else available / (1024 ** 3)
    allowed_gib = min(MAX_JOB_RSS_GIB, available_gib - MEMORY_HEADROOM_GIB) if available_gib is not None else None
    estimate_gib = largest_peak / (1024 ** 3)
    passed = allowed_gib is not None and allowed_gib > 0 and estimate_gib <= allowed_gib
    return {
        "status": "PASS_CSR_ESTIMATE" if passed else "UNKNOWN_OR_OVER_LIMIT",
        "method": "fit-free build_analyzer token counts; vocabulary cap and fit-only C2_B dimensions; conservative 64-bit CSR storage",
        "largest_estimated_simultaneous_csr_gib": estimate_gib,
        "available_physical_ram_gib": available_gib,
        "required_headroom_gib": MEMORY_HEADROOM_GIB,
        "job_rss_cap_gib": MAX_JOB_RSS_GIB,
        "effective_estimate_limit_gib": allowed_gib,
        "fit_free_preprocessing": True,
        "estimator_internal_working_memory": "UNKNOWN; no claim of total peak RSS from the matrix estimate alone",
        "folds": folds,
    }


def preflight() -> dict[str, Any]:
    """Read-only artifact/schema/cache/support validation; never fits or scores."""
    queue, card = _canonical_card()
    train_path, frame, y = _load_train()
    _, control_refs = _baseline_files_identity()
    manifest_sha = _sha256(BASELINE_MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("C09 current baseline manifest SHA differs from canonical queue")
    controls = {
        end: _load_control_fold(frame, y, end, control_refs)
        for end in FIT_ENDS
    }
    views: dict[str, dict[str, Any]] = {}
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    bundles: dict[str, dict[str, Any]] = {}
    for end, (low, high, name) in PRIMARY_BY_END.items():
        record = controls[end]
        positions = (cycle[record["rows"]] >= low) & (cycle[record["rows"]] <= high)
        bundles[name] = _bundle(record["rows"][positions], record["baseline"][positions],
                                record["baseline"][positions],
                                {key: values[positions] for key, values in record["labels"].items()},
                                record["edges"], cycle[record["rows"]][positions])
        views[name] = _support_only(y, bundles[name])
    for end, name in ((55, "H12_E"), (67, "H12_L")):
        record = controls[end]
        bundles[name] = _bundle(record["rows"], record["baseline"], record["baseline"],
                                record["labels"], record["edges"], cycle[record["rows"]])
        views[name] = _support_only(y, bundles[name])
    bundles["primary_pilot"] = _concat_bundles([bundles["W1"], bundles["W4"]])
    bundles["H12"] = _concat_bundles([bundles["H12_E"], bundles["H12_L"]])
    views["primary_pilot"] = _support_only(y, bundles["primary_pilot"])
    views["H12"] = _support_only(y, bundles["H12"])
    if views["primary_pilot"]["n"] != 9_064 or views["H12"]["n"] != 38_588:
        raise ValueError("C09 pilot primary/H12 support differs from canonical queue")
    identity = _input_identity(train_path, controls, {
        **control_refs,
        "manifest_sha256": manifest_sha,
    })
    memory_forecast = _sparse_memory_forecast(frame, controls)
    return {
        "status": ("PREFLIGHT_PASS_READ_ONLY_NO_FIT_NO_SCORE"
                   if memory_forecast["status"] == "PASS_CSR_ESTIMATE"
                   else "PREFLIGHT_BLOCKED_MEMORY_UNKNOWN_OR_OVER_LIMIT"),
        "card_id": "C09",
        "task": "classification",
        "train_sha256": _sha256(train_path),
        "train_rows": int(len(frame)),
        "train_cycle_range": [int(cycle.min()), int(cycle.max())],
        "target_support": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "baseline_manifest_sha256": manifest_sha,
        "baseline_primary_macro_f1_from_queue": queue["baseline"]["classification"]["primary"],
        "baseline_H12_macro_f1_from_queue": queue["baseline"]["classification"]["H12"],
        "fits": 0,
        "scores_computed": False,
        "candidate_metrics_computed": False,
        "target_use": "support denominators only; no candidate or comparative metric",
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
        "memory_forecast": memory_forecast,
        "folds": {
            str(end): {
                "fit_cycles": [47, end],
                "fit_rows": int(len(controls[end]["fit_rows"])),
                "score_cycles": [end + 1, HORIZON_BY_END[end]],
                "score_rows": int(len(controls[end]["rows"])),
                "c2b_control": controls[end]["identities"]["c2b"],
                "sparse_cache": controls[end]["identities"]["sparse"],
                "catboost_cache": controls[end]["identities"]["catboost"],
            }
            for end in FIT_ENDS
        },
        "support_only_views": views,
        "frozen_identity_sha256": digest_json(identity),
        "identity": identity,
        "output_paths_absent": {
            "recipe": not (CONFIG / "recipe.json").exists(),
            "freeze": not (CONFIG / "freeze.json").exists(),
            "results": not (REPORT / "results.json").exists(),
            "verification": not (REPORT / "verification.json").exists(),
            "summary": not (REPORT / "summary.md").exists(),
            "run_cache": not RUN_DIR.exists(),
        },
    }


def _write_freeze(preflight_result: dict[str, Any]) -> dict[str, Any]:
    if preflight_result.get("status") != "PREFLIGHT_PASS_READ_ONLY_NO_FIT_NO_SCORE":
        raise ValueError("A passing read-only preflight is required before C09 freeze")
    text_params = {
        key: ("numpy.float64" if key == "dtype" and str(value) == "<class 'numpy.float64'>" else value)
        for key, value in _TFIDF_PARAMS.items()
    }
    recipe = {
        "card_id": "C09",
        "task": "classification",
        "policy": "C_LR_REPLACE",
        "canonical_card_path": str(CARD.relative_to(ROOT)).replace("\\", "/"),
        "change": EXPECTED_CHANGE,
        "model": {
            "id": "C09",
            "base_model": "C2_B hybrid LogisticRegression",
            "input_feature_names": list(C09TextEntryFrictionClassifier().input_feature_names),
            "structured_and_text_preprocessing": "exact inherited C2_B fit-only transforms",
            "logistic_regression": dict(_LR_PARAMS),
            "tfidf": text_params,
            "interaction": "append CSR TF-IDF * I(parsed entry_friction == 1)",
            "interaction_reference_level": "entry_friction=0",
            "convergence_warning_policy": "fail the fit; no fallback estimator or parameter change",
        },
        "routing": {
            "stream_present": {"candidate_weight": 0.75, "frozen_sparse_binary_weight": 0.25},
            "stream_missing": "frozen CatBoost probability",
            "threshold": THRESHOLD,
        },
        "evaluation": {
            "pilot_fit_ends": list(FIT_ENDS),
            "fit_cycles": {str(end): [47, end] for end in FIT_ENDS},
            "score_horizons": {str(end): [end + 1, HORIZON_BY_END[end]] for end in FIT_ENDS},
            "primary_windows": {str(end): list(PRIMARY_BY_END[end][:2]) for end in FIT_ENDS},
            "primary_rows": 9064,
            "H12_rows": 38588,
            "gate": _canonical_card()[0]["evaluation"]["classification_gate"],
        },
        "source_and_input_identity_sha256": preflight_result["frozen_identity_sha256"],
        "source_and_input_identity": preflight_result["identity"],
        "memory_forecast": preflight_result["memory_forecast"],
        "training_source_boundary": "official classification Train only; no Test/Sample/Kaggle access",
    }
    recipe["recipe_sha256"] = digest_json(recipe)
    _save_or_verify_json(CONFIG / "recipe.json", recipe)
    freeze = {
        "status": "FROZEN_BEFORE_C09_PILOT_FITS",
        "signature": preflight_result["frozen_identity_sha256"],
        "identity": preflight_result["identity"],
        "recipe_sha256": recipe["recipe_sha256"],
        "recipe_file_sha256": _sha256(CONFIG / "recipe.json"),
        "official_fits_authorized_by_card": 2,
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
    }
    _save_json_exclusive(CONFIG / "freeze.json", freeze)
    return freeze


def _check_freeze() -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_result = preflight()
    if (preflight_result.get("status") != "PREFLIGHT_PASS_READ_ONLY_NO_FIT_NO_SCORE"
            or preflight_result.get("memory_forecast", {}).get("status") != "PASS_CSR_ESTIMATE"):
        raise ValueError("Current C09 preflight or memory forecast is not passing; refusing to run")
    freeze = _read_json(CONFIG / "freeze.json")
    if freeze.get("identity") != preflight_result["identity"]:
        raise ValueError("C09 frozen source/input identity changed")
    if freeze.get("signature") != preflight_result["frozen_identity_sha256"]:
        raise ValueError("C09 freeze signature mismatch")
    recipe = _read_json(CONFIG / "recipe.json")
    if (freeze.get("recipe_file_sha256") != _sha256(CONFIG / "recipe.json")
            or freeze.get("recipe_sha256") != recipe.get("recipe_sha256")):
        raise ValueError("C09 frozen recipe file/hash mismatch")
    return freeze, preflight_result


def _reuse_verified_fit(record: dict[str, Any], freeze_signature: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    end = int(record["fit_end"])
    cache = RUN_DIR / f"fit{end}.npz"
    sidecar = cache.with_suffix(".json")
    attempt_path = RUN_DIR / f"fit{end}.attempt.json"
    if cache.exists() != sidecar.exists():
        raise ValueError(f"Partial C09 cache/sidecar pair is incomplete at cutoff {end}")
    if not cache.exists():
        if attempt_path.exists():
            raise ValueError(f"C09 fit attempt receipt exists without a verified cache at cutoff {end}")
        return None
    metadata = _read_json(sidecar)
    if metadata.get("freeze_signature") != freeze_signature or metadata.get("npz_sha256") != _sha256(cache):
        raise ValueError(f"Partial C09 cache identity/hash mismatch at cutoff {end}")
    with np.load(cache, allow_pickle=False) as archive:
        rows = np.asarray(archive["rows"], dtype=np.int64).copy()
        raw = np.asarray(archive["candidate_model_probability"], dtype=np.float64).copy()
        routed = np.asarray(archive["candidate_routed_probability"], dtype=np.float64).copy()
        baseline = np.asarray(archive["baseline_probability"], dtype=np.float64).copy()
    if not np.array_equal(rows, record["rows"]) or not (raw.shape == routed.shape == baseline.shape == rows.shape):
        raise ValueError(f"Partial C09 cache row mapping/shape mismatch at cutoff {end}")
    if not np.isfinite(raw).all() or not np.isfinite(routed).all() or not np.isfinite(baseline).all():
        raise ValueError(f"Partial C09 cache contains non-finite probabilities at cutoff {end}")
    if ((raw < 0) | (raw > 1)).any() or ((routed < 0) | (routed > 1)).any() or ((baseline < 0) | (baseline > 1)).any():
        raise ValueError(f"Partial C09 cache probabilities are out of range at cutoff {end}")
    if not np.array_equal(baseline, record["baseline"]):
        raise ValueError(f"Partial C09 baseline changed at cutoff {end}")
    expected_routed = np.where(
        record["missing"], record["cb"],
        PRESENT_C09_WEIGHT * raw + PRESENT_SPARSE_WEIGHT * record["sparse_raw"],
    )
    if not np.array_equal(routed, expected_routed):
        raise ValueError(f"Partial C09 route changed at cutoff {end}")
    if not np.array_equal(routed[record["missing"]], record["cb"][record["missing"]]):
        raise ValueError(f"Partial C09 missing branch changed at cutoff {end}")
    if array_digest(raw) != metadata.get("candidate_model_probability_sha256"):
        raise ValueError(f"Partial C09 raw candidate prediction hash mismatch at cutoff {end}")
    if array_digest(routed) != metadata.get("candidate_routed_probability_sha256"):
        raise ValueError(f"Partial C09 routed prediction hash mismatch at cutoff {end}")
    if not attempt_path.is_file() or metadata.get("fit_attempt_receipt_sha256") != _sha256(attempt_path):
        raise ValueError(f"Partial C09 fit-attempt receipt is absent or changed at cutoff {end}")
    attempt = _read_json(attempt_path)
    if (attempt.get("status") != "COMPLETED" or attempt.get("fit_end") != end
            or int(attempt.get("convergence_warning_count", -1)) != 0):
        raise ValueError(f"Partial C09 fit-attempt receipt is not a clean completed fit at cutoff {end}")
    state = metadata.get("model_state")
    if not isinstance(state, dict) or state.get("id") != "C09" or state.get("fit_rows") != int(record["fit_rows"].size):
        raise ValueError(f"Partial C09 model-state identity mismatch at cutoff {end}")
    return raw, routed, metadata


def _manual_macro_f1(actual: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(actual, dtype=np.int8)
    predicted = (np.asarray(probability, dtype=float) >= THRESHOLD).astype(np.int8)
    values = []
    for positive in (0, 1):
        tp = int(np.count_nonzero((labels == positive) & (predicted == positive)))
        fp = int(np.count_nonzero((labels != positive) & (predicted == positive)))
        fn = int(np.count_nonzero((labels == positive) & (predicted != positive)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(values))


def _independent_group_labels(
    frame: pd.DataFrame,
    rows: np.ndarray,
    controls: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Rebuild group masks directly from frozen Train rows and fit-only edges."""
    subset = frame.iloc[rows]
    values, missing, invalid = _number_array(subset["entry_friction"])
    known = (missing == 0) & (invalid == 0) & np.isfinite(values)
    ef0 = known & (values == 0.0)
    ef1 = known & (values == 1.0)
    ef_labels = np.where(ef0, "EF0", np.where(ef1, "EF1", "EF_UNKNOWN"))
    stream_labels = np.where(stream_missing_mask(subset), "missing", "present")

    cycle = pd.to_numeric(subset["cycle_code"], errors="raise").to_numpy(dtype=int)
    text_values, text_missing, text_invalid = _number_array(subset["text_load"])
    text_labels = np.full(len(rows), "missing", dtype=object)
    valid = (text_missing == 0) & (text_invalid == 0) & np.isfinite(text_values)
    for end, (low, high) in ((55, (56, 67)), (67, (68, 79))):
        local = (cycle >= low) & (cycle <= high) & valid
        edges = np.asarray(controls[end]["edges"]["text_load"], dtype=float)
        if not local.any():
            continue
        if len(edges) == 0:
            continue
        if len(edges) == 1:
            text_labels[local] = "Q1"
        else:
            bins = np.searchsorted(edges[1:-1], text_values[local], side="left")
            text_labels[local] = np.asarray([f"Q{int(index) + 1}" for index in bins], dtype=object)
    return {"entry_friction": ef_labels, "stream": stream_labels, "text_load": text_labels}


def _manual_positive_recall(actual: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(actual, dtype=np.int8)
    predicted = np.asarray(probability, dtype=float) >= THRESHOLD
    positives = labels == 1
    support = int(positives.sum())
    return 0.0 if support == 0 else float(np.count_nonzero(predicted & positives) / support)


def _independent_group_guard_audit(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: dict[str, np.ndarray],
    published: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Recompute support, metrics, and card guards from direct row masks."""
    audits: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    checks["group axis schema matches"] = set(labels) == set(published)
    for axis, axis_labels in labels.items():
        if axis not in published:
            checks[f"{axis} published group axis exists"] = False
            continue
        direct_group_names = set(str(value) for value in axis_labels)
        checks[f"{axis} exact group names match"] = direct_group_names == set(published[axis])
        audits[axis] = {}
        for group in sorted(direct_group_names):
            mask = np.asarray([str(value) == group for value in axis_labels], dtype=bool)
            yy = actual[mask]
            base = baseline[mask]
            pred = candidate[mask]
            negatives = int(np.count_nonzero(yy == 0))
            positives = int(np.count_nonzero(yy == 1))
            supported = (
                int(mask.sum()) >= int(gate["supported_group_n_min"])
                and min(negatives, positives) >= int(gate["positive_and_negative_per_group_min"])
            )
            base_f1 = _manual_macro_f1(yy, base)
            candidate_f1 = _manual_macro_f1(yy, pred)
            base_recall = _manual_positive_recall(yy, base)
            candidate_recall = _manual_positive_recall(yy, pred)
            f1_harm = base_f1 - candidate_f1
            recall_harm = base_recall - candidate_recall
            expected_gate = (
                bool(f1_harm <= float(gate["supported_group_F1_harm_max"])
                     and recall_harm <= float(gate["positive_class_recall_harm_max"]))
                if supported else None
            )
            published_group = published[axis].get(group)
            matching = isinstance(published_group, dict)
            if matching:
                matching = (
                    published_group.get("n") == int(mask.sum())
                    and published_group.get("negative_support") == negatives
                    and published_group.get("positive_support") == positives
                    and published_group.get("supported") is supported
                    and published_group.get("gate_pass") == expected_gate
                )
                for key, value in (
                    ("baseline_macro_f1", base_f1),
                    ("candidate_macro_f1", candidate_f1),
                    ("macro_f1_harm", f1_harm),
                    ("baseline_positive_recall", base_recall),
                    ("candidate_positive_recall", candidate_recall),
                    ("positive_recall_harm", recall_harm),
                ):
                    observed = published_group.get(key)
                    matching = matching and observed is not None and abs(float(observed) - value) <= 1e-12
            checks[f"{axis}:{group} direct denominator and guard match"] = bool(matching)
            audits[axis][group] = {
                "n": int(mask.sum()),
                "negative_support": negatives,
                "positive_support": positives,
                "supported": bool(supported),
                "baseline_macro_f1": base_f1,
                "candidate_macro_f1": candidate_f1,
                "macro_f1_harm": f1_harm,
                "baseline_positive_recall": base_recall,
                "candidate_positive_recall": candidate_recall,
                "positive_recall_harm": recall_harm,
                "gate_pass": expected_gate,
            }
    return audits, checks


def _verify_pilot_result(
    result: dict[str, Any],
    bundles: dict[str, dict[str, Any]],
    controls: dict[int, dict[str, Any]],
    y: np.ndarray,
    frame: pd.DataFrame,
    freeze_signature: str,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["two pilot cutoffs accounted"] = set(result["fit_states"]) == {"55", "67"}
    checks["fit attempts contain no convergence warnings"] = (
        result["fit_attempt_accounting"]["convergence_warning_count"] == 0
    )
    checks["exact primary and H12 denominators"] = (
        result["views"]["primary_pilot"]["n"] == 9064 and result["views"]["H12"]["n"] == 38588
    )
    checks["baseline queue metrics reconstruct"] = (
        abs(result["control"]["baseline_macro_f1_row_reconstructed"]["primary"]
            - result["control"]["baseline_macro_f1_from_queue"]["primary"]) <= 1e-10
        and abs(result["control"]["baseline_macro_f1_row_reconstructed"]["H12"]
                - result["control"]["baseline_macro_f1_from_queue"]["H12"]) <= 1e-10
    )

    persisted_by_row: dict[int, tuple[float, float, float]] = {}
    cache_checks: dict[str, Any] = {}
    route_checks: dict[str, bool] = {}
    for end, record in controls.items():
        cache_path = RUN_DIR / f"fit{end}.npz"
        sidecar_path = cache_path.with_suffix(".json")
        attempt_path = RUN_DIR / f"fit{end}.attempt.json"
        metadata = _read_json(sidecar_path)
        attempt = _read_json(attempt_path)
        npz_sha = _sha256(cache_path)
        checks[f"fit{end} NPZ checksum matches sidecar"] = metadata.get("npz_sha256") == npz_sha
        checks[f"fit{end} frozen signature matches"] = metadata.get("freeze_signature") == freeze_signature
        checks[f"fit{end} attempt receipt hash matches"] = (
            metadata.get("fit_attempt_receipt_sha256") == _sha256(attempt_path)
        )
        checks[f"fit{end} attempt receipt is completed and warning-free"] = (
            attempt.get("status") == "COMPLETED"
            and attempt.get("freeze_signature") == freeze_signature
            and attempt.get("fit_rows") == int(record["fit_rows"].size)
            and attempt.get("fit_cycles") == [47, end]
            and attempt.get("score_cycles") == [end + 1, HORIZON_BY_END[end]]
            and int(attempt.get("convergence_warning_count", -1)) == 0
            and attempt.get("model_fit_attempt", {}).get("status") == "FIT_SUCCESS"
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            rows = np.asarray(archive["rows"], dtype=np.int64).copy()
            raw = np.asarray(archive["candidate_model_probability"], dtype=np.float64).copy()
            routed = np.asarray(archive["candidate_routed_probability"], dtype=np.float64).copy()
            baseline = np.asarray(archive["baseline_probability"], dtype=np.float64).copy()
        checks[f"fit{end} exact row map and vector shapes"] = (
            np.array_equal(rows, record["rows"])
            and raw.shape == routed.shape == baseline.shape == rows.shape
        )
        checks[f"fit{end} prediction probabilities are finite and bounded"] = bool(
            np.isfinite(raw).all() and np.isfinite(routed).all() and np.isfinite(baseline).all()
            and ((raw >= 0) & (raw <= 1)).all()
            and ((routed >= 0) & (routed <= 1)).all()
            and ((baseline >= 0) & (baseline <= 1)).all()
        )
        checks[f"fit{end} model state and fit boundary match"] = (
            metadata.get("fit_rows") == int(record["fit_rows"].size)
            and metadata.get("fit_cycles") == [47, end]
            and metadata.get("score_cycles") == [end + 1, HORIZON_BY_END[end]]
            and metadata.get("model_state", {}).get("id") == "C09"
            and metadata.get("model_state", {}).get("fit_rows") == int(record["fit_rows"].size)
            and result["fit_states"][str(end)]["fit_rows"] == int(record["fit_rows"].size)
        )
        checks[f"fit{end} saved prediction hashes match sidecar"] = (
            array_digest(raw) == metadata.get("candidate_model_probability_sha256")
            and array_digest(routed) == metadata.get("candidate_routed_probability_sha256")
            and array_digest(rows.astype(np.float64)) == metadata.get("rows_sha256")
        )
        expected_route = np.where(
            record["missing"],
            record["cb"],
            PRESENT_C09_WEIGHT * raw + PRESENT_SPARSE_WEIGHT * record["sparse_raw"],
        )
        route_checks[str(end)] = bool(
            np.isfinite(raw).all() and np.isfinite(routed).all() and np.isfinite(baseline).all()
            and np.array_equal(baseline, record["baseline"])
            and np.array_equal(routed, expected_route)
            and np.array_equal(routed[record["missing"]], record["cb"][record["missing"]])
            and np.array_equal(record["candidate"], routed)
        )
        checks[f"fit{end} present and missing routes independently reconstruct"] = route_checks[str(end)]
        checks[f"fit{end} fit and score rows are disjoint"] = (
            np.intersect1d(record["fit_rows"], rows).size == 0
            and np.array_equal(rows, _expected_rows(frame, end, HORIZON_BY_END[end]))
        )
        for row, raw_value, routed_value, base_value in zip(rows, raw, expected_route, baseline):
            row_int = int(row)
            if row_int in persisted_by_row:
                checks["saved score row maps are globally unique"] = False
            else:
                persisted_by_row[row_int] = (float(raw_value), float(routed_value), float(base_value))
        cache_checks[str(end)] = {
            "npz_path": str(cache_path.relative_to(ROOT)).replace("\\", "/"),
            "npz_sha256": npz_sha,
            "sidecar_path": str(sidecar_path.relative_to(ROOT)).replace("\\", "/"),
            "attempt_receipt_path": str(attempt_path.relative_to(ROOT)).replace("\\", "/"),
            "attempt_receipt_sha256": _sha256(attempt_path),
        }

    checks.setdefault("saved score row maps are globally unique", True)
    manual_checks: dict[str, dict[str, float]] = {}
    group_audits: dict[str, Any] = {}
    group_gate_checks: dict[str, bool] = {}
    card_gate = _canonical_card()[0]["evaluation"]["classification_gate"]
    for name, bundle in bundles.items():
        rows = bundle["rows"]
        row_records = [persisted_by_row.get(int(row)) for row in rows]
        aligned = all(value is not None for value in row_records)
        checks[f"{name} saved rows cover exact view"] = aligned
        if not aligned:
            continue
        candidate = np.asarray([value[1] for value in row_records], dtype=float)
        baseline = np.asarray([value[2] for value in row_records], dtype=float)
        actual = y[rows]
        observed = result["views"][name]
        checks[f"{name} persisted arrays reproduce Macro F1"] = (
            abs(_manual_macro_f1(actual, baseline) - observed["baseline"]["macro_f1"]) <= 1e-12
            and abs(_manual_macro_f1(actual, candidate) - observed["candidate"]["macro_f1"]) <= 1e-12
        )
        checks[f"{name} persisted baseline/candidate equal scored bundle"] = (
            np.array_equal(baseline, bundle["baseline"])
            and np.array_equal(candidate, bundle["candidate"])
        )
        checks[f"{name} source rows unique"] = len(np.unique(rows)) == len(rows)
        manual_checks[name] = {
            "baseline_macro_f1": _manual_macro_f1(actual, baseline),
            "candidate_macro_f1": _manual_macro_f1(actual, candidate),
        }
        independent_labels = _independent_group_labels(frame, rows, controls)
        checks[f"{name} group masks independently rebuilt"] = all(
            axis in bundle["labels"] and np.array_equal(independent_labels[axis], bundle["labels"][axis])
            for axis in independent_labels
        )
        group_audits[name], checks_for_groups = _independent_group_guard_audit(
            actual, baseline, candidate, independent_labels,
            observed["group_metrics"], card_gate,
        )
        group_gate_checks.update({f"{name}:{key}": value for key, value in checks_for_groups.items()})
    checks.update(group_gate_checks)
    checks["fit and score rows disjoint; missing-stream fallback exact"] = all(route_checks.values())
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "manual_macro_f1_from_reloaded_prediction_caches": manual_checks,
        "independent_group_denominators_and_guards": group_audits,
        "per_cutoff_route_and_boundary_checks": route_checks,
        "prediction_cache_and_receipt_hashes": cache_checks,
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
    }


def _write_summary(result: dict[str, Any], verification: dict[str, Any]) -> None:
    lines = [
        "# C09 — Pilot summary",
        "",
        f"- Verdict: **{result.get('verdict', result.get('status'))}**",
        f"- Status: `{result.get('status')}`",
        f"- Frozen identity: `{result.get('freeze_signature', 'not frozen')}`",
        f"- Verification: `{verification.get('status')}`",
        f"- Predictor fits: {result.get('fit_attempt_accounting', {}).get('successful_predictor_fits', 0)} successful / "
        f"{result.get('fit_attempt_accounting', {}).get('attempts_started', 0)} attempts; convergence warnings: "
        f"{result.get('fit_attempt_accounting', {}).get('convergence_warning_count', 0)}",
        "",
    ]
    if result.get("views"):
        lines.extend([
            "| View | n | Baseline Macro F1 | C09 Macro F1 | Delta |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in ("W1", "W4", "primary_pilot", "H12_E", "H12_L", "H12"):
            view = result["views"][name]
            lines.append(
                f"| {name} | {view['n']} | {view['baseline']['macro_f1']:.6f} | "
                f"{view['candidate']['macro_f1']:.6f} | {view['delta_macro_f1']:+.6f} |"
            )
        lines.extend(["", "Supported group harms and denominators are in `results.json`; small groups remain descriptive.", ""])
    else:
        lines.extend(["- Candidate metrics: not computed because a predictor fit failed.", ""])
    lines.extend([
        "- Route: present stream uses 0.75 C09 + 0.25 frozen sparse binary; missing stream uses frozen CatBoost; threshold 0.50.",
        "- Limitation: temporal Train-cycle development evidence, not an untouched holdout.",
        "- Test/Sample/Kaggle access: none.",
    ])
    _write_exclusive_atomic(SUMMARY_PATH, ("\n".join(lines) + "\n").encode("utf-8"))


def _reserve_c09_budget(ledger_path: Path) -> dict[str, Any]:
    queue, card = _canonical_card()
    preset = queue.get("execution_preset", {})
    if int(preset.get("concurrent_fit_jobs", -1)) != 1 or int(preset.get("threads_per_job", -1)) != 2:
        raise ValueError("C09 requires one serialized training job with two threads")
    if int(card.get("pilot_predictor_fits", -1)) != 2:
        raise ValueError("C09 full pilot fit reservation differs from canonical card")
    cap = int(preset.get("max_successful_new_predictor_fits", -1))
    if cap <= 0:
        raise ValueError("C09 campaign fit cap is absent or invalid")
    return reserve_card_budget(
        ledger_path,
        budget_key="C09:PILOT",
        card_id="C09",
        predictor_fits=int(card["pilot_predictor_fits"]),
        cap=cap,
    )


def _run_pilot() -> dict[str, Any]:
    freeze, preflight_result = _check_freeze()
    if any(path.exists() for path in (RESULTS_PATH, VERIFICATION_PATH, SUMMARY_PATH)):
        raise FileExistsError("A C09 terminal report already exists; refusing to overwrite")
    with exclusive_training_slot(TRAINING_LOCK_PATH):
        locked_freeze, locked_preflight = _check_freeze()
        if locked_freeze["signature"] != freeze["signature"]:
            raise ValueError("C09 frozen identity changed while acquiring the training slot")
        reservation = _reserve_c09_budget(BATCH_LEDGER_PATH)
        return _run_pilot_locked(locked_freeze, locked_preflight, reservation)


def _run_pilot_locked(
    freeze: dict[str, Any],
    preflight_result: dict[str, Any],
    budget_reservation: dict[str, Any],
) -> dict[str, Any]:
    _, frame, y = _load_train()
    _, control_refs = _baseline_files_identity()
    controls = {end: _load_control_fold(frame, y, end, control_refs) for end in FIT_ENDS}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    candidate_bundles: dict[str, dict[str, Any]] = {}
    fit_states: dict[str, Any] = {}
    started = perf_counter()
    new_fits = 0
    reused_fits = 0
    fit_attempts: list[dict[str, Any]] = []
    failed_fit: dict[str, Any] | None = None
    for end in FIT_ENDS:
        record = controls[end]
        reused = _reuse_verified_fit(record, freeze["signature"])
        if reused is not None:
            prediction, routed, metadata = reused
            attempt_path = RUN_DIR / f"fit{end}.attempt.json"
            state = metadata["model_state"]
            fit_states[str(end)] = {
                "fit_rows": int(record["fit_rows"].size),
                "fit_cycles": [47, end],
                "score_cycles": [end + 1, HORIZON_BY_END[end]],
                "cache_status": "REUSED_HASH_VERIFIED",
                "cache_sha256": metadata["npz_sha256"],
                "seconds": float(metadata["seconds"]),
                "state": state,
            }
            reused_fits += 1
            fit_attempts.append(_read_json(attempt_path))
        else:
            model = C09TextEntryFrictionClassifier()
            fields = list(model.input_feature_names)
            fit_mask = pd.Series(cycles).between(47, end).to_numpy()
            fit_start = perf_counter()
            attempt_path = RUN_DIR / f"fit{end}.attempt.json"
            attempt_receipt = {
                "card_id": "C09",
                "fit_end": end,
                "status": "RUNNING",
                "attempt_number": 1,
                "fit_rows": int(fit_mask.sum()),
                "fit_cycles": [47, end],
                "score_cycles": [end + 1, HORIZON_BY_END[end]],
                "source_train_sha256": EXPECTED_TRAIN_SHA256,
                "freeze_signature": freeze["signature"],
                "convergence_warning_count": 0,
                "test_read": False,
                "sample_read": False,
                "kaggle_access": False,
            }
            _save_json_exclusive(attempt_path, attempt_receipt)
            try:
                with threadpool_limits(limits=2):
                    model.fit(frame.loc[fit_mask, fields], y[fit_mask])
                    state = model.audit_state()
                    prediction = model.predict_proba(frame.iloc[record["rows"]][fields])
            except Exception as exc:
                failed_model_attempt = dict(model._c09_fit_attempt)
                attempt_receipt.update({
                    "status": "FAILED",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "convergence_warning_count": int(failed_model_attempt.get("convergence_warning_count", 0)),
                    "model_fit_attempt": failed_model_attempt,
                    "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "seconds": float(perf_counter() - fit_start),
                })
                _save_json_atomic_replace(attempt_path, attempt_receipt)
                fit_attempts.append(attempt_receipt)
                failed_fit = {
                    "fit_end": end,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "convergence_warning_count": int(failed_model_attempt.get("convergence_warning_count", 0)),
                    "fit_attempt_receipt": str(attempt_path.relative_to(ROOT)).replace("\\", "/"),
                }
                break
            fit_seconds = float(perf_counter() - fit_start)
            if state != model.audit_state():
                raise AssertionError(f"C09 model state changed after prediction at cutoff {end}")
            if prediction.shape != record["rows"].shape or not np.isfinite(prediction).all():
                raise ValueError(f"C09 prediction output invalid at cutoff {end}")
            if ((prediction < 0) | (prediction > 1)).any():
                raise ValueError(f"C09 prediction probabilities outside [0,1] at cutoff {end}")
            attempt_receipt.update({
                "status": "COMPLETED",
                "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "seconds": fit_seconds,
                "convergence_warning_count": int(state["c09_interaction"]["fit_attempt"]["convergence_warning_count"]),
                "model_fit_attempt": state["c09_interaction"]["fit_attempt"],
                "model_state_sha256": state["model_state_sha256"],
            })
            _save_json_atomic_replace(attempt_path, attempt_receipt)
            fit_attempts.append(attempt_receipt)
            routed = np.where(
                record["missing"],
                record["cb"],
                PRESENT_C09_WEIGHT * prediction + PRESENT_SPARSE_WEIGHT * record["sparse_raw"],
            )
            if not np.array_equal(routed[record["missing"]], record["cb"][record["missing"]]):
                raise AssertionError(f"C09 missing-stream route changed at cutoff {end}")
            cache_payload = {
                "rows": record["rows"].astype(np.int64),
                "candidate_model_probability": prediction.astype(np.float64),
                "candidate_routed_probability": routed.astype(np.float64),
                "baseline_probability": record["baseline"].astype(np.float64),
            }
            cache_path = RUN_DIR / f"fit{end}.npz"
            _save_npz_exclusive(cache_path, **cache_payload)
            metadata = {
                "freeze_signature": freeze["signature"],
                "npz_sha256": _sha256(cache_path),
                "fit_attempt_receipt_sha256": _sha256(attempt_path),
                "candidate_model_probability_sha256": array_digest(prediction),
                "candidate_routed_probability_sha256": array_digest(routed),
                "rows_sha256": array_digest(record["rows"].astype(np.float64)),
                "fit_rows": int(fit_mask.sum()),
                "fit_cycles": [47, end],
                "score_cycles": [end + 1, HORIZON_BY_END[end]],
                "seconds": fit_seconds,
                "model_state": state,
                "test_read": False,
                "sample_read": False,
                "kaggle_access": False,
            }
            _save_json_exclusive(cache_path.with_suffix(".json"), metadata)
            fit_states[str(end)] = {
                "fit_rows": int(fit_mask.sum()),
                "fit_cycles": [47, end],
                "score_cycles": [end + 1, HORIZON_BY_END[end]],
                "cache_status": "NEW_FIT",
                "cache_sha256": metadata["npz_sha256"],
                "seconds": fit_seconds,
                "state": state,
            }
            new_fits += 1
        record["candidate"] = routed
        labels = record["labels"]
        full_bundle = _bundle(record["rows"], record["baseline"], routed, labels, record["edges"],
                              cycles[record["rows"]])
        key = "H12_E" if end == 55 else "H12_L"
        candidate_bundles[key] = full_bundle
        low, high, window_name = PRIMARY_BY_END[end]
        keep = (cycles[record["rows"]] >= low) & (cycles[record["rows"]] <= high)
        candidate_bundles[window_name] = _subset_bundle(full_bundle, keep)
    if failed_fit is not None:
        accounting = {
            "attempts_started": len(fit_attempts),
            "successful_predictor_fits": int(sum(item.get("status") == "COMPLETED" for item in fit_attempts)),
            "failed_predictor_attempts": int(sum(item.get("status") == "FAILED" for item in fit_attempts)),
            "convergence_warning_count": int(sum(int(item.get("convergence_warning_count", 0)) for item in fit_attempts)),
            "fit_attempts": fit_attempts,
        }
        failure_result = {
            "status": "PILOT_REWORK_CONVERGENCE_OR_FIT_FAILURE",
            "verdict": "REWORK",
            "experiment_id": "optimization_v3_batch1_C09",
            "freeze_signature": freeze["signature"],
            "campaign_budget_reservation": budget_reservation,
            "failed_fit": failed_fit,
            "fit_attempt_accounting": accounting,
            "new_predictor_fits": new_fits,
            "reused_verified_fits": reused_fits,
            "fit_states": fit_states,
            "candidate_metrics_computed": False,
            "primary_denominator": 9064,
            "H12_denominator": 38588,
            "test_read": False,
            "sample_read": False,
            "kaggle_access": False,
            "interpretation_limit": "No candidate comparison is available because the declared fit did not complete.",
        }
        failure_verification = {
            "status": "REWORK_FIT_FAILED_NO_CANDIDATE_METRICS",
            "failed_fit": failed_fit,
            "attempt_receipts_verified": all(
                (RUN_DIR / f"fit{item['fit_end']}.attempt.json").is_file()
                and _read_json(RUN_DIR / f"fit{item['fit_end']}.attempt.json").get("status") == item["status"]
                for item in fit_attempts
            ),
            "candidate_metrics_computed": False,
            "test_read": False,
            "sample_read": False,
            "kaggle_access": False,
        }
        REPORT.mkdir(parents=True, exist_ok=True)
        _save_json_exclusive(RESULTS_PATH, failure_result)
        _save_json_exclusive(VERIFICATION_PATH, failure_verification)
        _write_summary(failure_result, failure_verification)
        return failure_result
    primary = _concat_bundles([candidate_bundles["W1"], candidate_bundles["W4"]])
    h12 = _concat_bundles([candidate_bundles["H12_E"], candidate_bundles["H12_L"]])
    views = {
        name: _evaluate_bundle(y, bundle)
        for name, bundle in {
            **candidate_bundles,
            "primary_pilot": primary,
            "H12": h12,
        }.items()
    }
    canonical_baseline = _canonical_card()[0]["baseline"]["classification"]
    if abs(views["primary_pilot"]["baseline"]["macro_f1"] - canonical_baseline["primary"]) > 1e-10:
        raise ValueError("Row-reconstructed C09 primary control differs from the frozen baseline metric")
    if abs(views["H12"]["baseline"]["macro_f1"] - canonical_baseline["H12"]) > 1e-10:
        raise ValueError("Row-reconstructed C09 H12 control differs from the frozen baseline metric")
    gates = _canonical_card()[0]["evaluation"]["classification_gate"]
    group_gate_pass = all(
        record.get("gate_pass") is not False
        for view in views.values()
        for groups in view["group_metrics"].values()
        for record in groups.values()
        if record.get("supported")
    )
    primary_gain = views["primary_pilot"]["delta_macro_f1"]
    h12_harm = -views["H12"]["delta_macro_f1"]
    individual_harms = {
        name: -views[name]["delta_macro_f1"] for name in ("W1", "W4")
    }
    pilot_gate = {
        "primary_gain_min": float(gates["pilot_primary_gain_min"]),
        "primary_gain": float(primary_gain),
        "primary_pass": bool(primary_gain >= float(gates["pilot_primary_gain_min"])),
        "H12_harm_max": float(gates["pilot_H12_harm_max"]),
        "H12_harm": float(h12_harm),
        "H12_pass": bool(h12_harm <= float(gates["pilot_H12_harm_max"])),
        "individual_view_harm_max": float(gates["individual_pilot_view_harm_max"]),
        "individual_view_harms": individual_harms,
        "individual_views_pass": bool(all(
            harm <= float(gates["individual_pilot_view_harm_max"]) for harm in individual_harms.values()
        )),
        "supported_group_gates_pass": bool(group_gate_pass),
    }
    pilot_gate["pass"] = bool(
        pilot_gate["primary_pass"] and pilot_gate["H12_pass"]
        and pilot_gate["individual_views_pass"] and pilot_gate["supported_group_gates_pass"]
    )
    result = {
        "status": "PILOT_COMPLETE_ADVANCE" if pilot_gate["pass"] else "PILOT_COMPLETE_RETIRE",
        "verdict": "ADVANCE" if pilot_gate["pass"] else "RETIRE",
        "experiment_id": "optimization_v3_batch1_C09",
        "freeze_signature": freeze["signature"],
        "campaign_budget_reservation": budget_reservation,
        "new_predictor_fits": new_fits,
        "reused_verified_fits": reused_fits,
        "fit_attempt_accounting": {
            "attempts_started": len(fit_attempts),
            "successful_predictor_fits": int(sum(item.get("status") == "COMPLETED" for item in fit_attempts)),
            "failed_predictor_attempts": int(sum(item.get("status") == "FAILED" for item in fit_attempts)),
            "convergence_warning_count": int(sum(int(item.get("convergence_warning_count", 0)) for item in fit_attempts)),
            "fit_attempts": fit_attempts,
        },
        "score_threshold": THRESHOLD,
        "primary_denominator": int(views["primary_pilot"]["n"]),
        "H12_denominator": int(views["H12"]["n"]),
        "fallback_coverage": {
            name: {
                "stream_missing_n": view["group_metrics"]["stream"]["missing"]["n"],
                "stream_present_n": view["group_metrics"]["stream"]["present"]["n"],
            }
            for name, view in views.items()
        },
        "memory_forecast": preflight_result["memory_forecast"],
        "test_read": False,
        "sample_read": False,
        "kaggle_access": False,
        "elapsed_seconds": float(perf_counter() - started),
        "fit_states": fit_states,
        "views": views,
        "pilot_gate": pilot_gate,
        "control": {
            "present": ".75 C2_B + .25 frozen sparse binary; candidate substitutes C09 for C2_B",
            "missing": "frozen CatBoost, unchanged",
            "baseline_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "baseline_macro_f1_from_queue": {
                "primary": canonical_baseline["primary"],
                "H12": canonical_baseline["H12"],
            },
            "baseline_macro_f1_row_reconstructed": {
                "primary": views["primary_pilot"]["baseline"]["macro_f1"],
                "H12": views["H12"]["baseline"]["macro_f1"],
            },
        },
        "interpretation_limit": "Repeated Train-cycle temporal development evidence; not an untouched holdout.",
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    verification = _verify_pilot_result(
        result,
        {**candidate_bundles, "primary_pilot": primary, "H12": h12},
        controls,
        y,
        frame,
        freeze["signature"],
    )
    if verification["status"] != "PASS":
        result["status"] = "PILOT_VERIFICATION_FAILED"
        result["verdict"] = "REWORK"
    _save_json_exclusive(RESULTS_PATH, result)
    _save_json_exclusive(VERIFICATION_PATH, verification)
    _write_summary(result, verification)
    return result


def run_card(
    card: dict[str, Any],
    queue: dict[str, Any],
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Generic dispatcher bridge; official C09 runs stay on explicit CLI modes."""
    canonical_queue, canonical_card = _canonical_card()
    if queue != canonical_queue or card != canonical_card:
        raise BatchIntegrityError("C09 dispatcher received a noncanonical card or queue")
    if preflight_only:
        return preflight()
    raise BatchIntegrityError("C09 official execution requires explicit --write-freeze followed by --run")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true", help="read-only checks; no fitting or scoring")
    mode.add_argument("--write-freeze", action="store_true", help="persist pre-score recipe/input freeze")
    mode.add_argument("--run", action="store_true", help="run the two official temporal pilot fits after freeze")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2, allow_nan=False))
        return 0
    if args.write_freeze:
        frozen = _write_freeze(preflight())
        print(json.dumps({"status": frozen["status"], "signature": frozen["signature"]}, indent=2))
        return 0
    if args.run:
        print(json.dumps(_run_pilot(), indent=2, allow_nan=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
