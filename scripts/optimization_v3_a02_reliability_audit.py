"""Zero-fit reliability and temporal mechanism audit for frozen A02 outputs.

This script loads persisted A02 models only to read their fitted bank state and
transform query text for a zero-vector flag. It never calls a fit method or
predict method, never reads Test/SampleSubmission, and never edits probabilities.
Run from the repository root with:

    python -B scripts/optimization_v3_a02_reliability_audit.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_a02_neighbors import (  # noqa: E402
    ResidualNeighborPredictor,
    permute_residuals_within_cycle_probability_decile,
)

TRAIN_REL = "data/raw/classification/Train.csv"
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
PILOT_PROTOCOL_REL = "experiments/optimization_v3/c19_a02_v2/A02/protocol.json"
PILOT_FREEZE_REL = "experiments/optimization_v3/c19_a02_v2/A02/freeze.json"
CONT_PROTOCOL_REL = "experiments/optimization_v3/a02_continuation_v1/protocol.json"
CONT_FREEZE_REL = "experiments/optimization_v3/a02_continuation_v1/freeze_v2.json"
PILOT_REPORT = "reports/optimization_v3/a02_neighbors_v1"
CONT_REPORT = "reports/optimization_v3/a02_continuation_v1"
PLAN_REL = "reports/optimization_v3/a02_reliability_audit_v1/AUDIT_PLAN.md"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_reliability_audit_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_a02_reliability_audit_v1"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"

CACHE_PATHS = {
    55: "experiments/runs/optimization_v3_batch1/C16/C16_fit55.npz",
    59: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit59.npz",
    63: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit63.npz",
    67: "experiments/runs/optimization_v3_batch1/C16/C16_fit67.npz",
    71: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit71.npz",
    75: "experiments/runs/optimization_v3_final_continuation/C16/C16_fit75.npz",
}
CACHE_WINDOWS = {55: (56, 67), 59: (60, 63), 63: (64, 67),
                 67: (68, 79), 71: (72, 75), 75: (76, 79)}
ORIGIN_FIRST4 = {59: (60, 63), 63: (64, 67), 67: (68, 71),
                 71: (72, 75), 75: (76, 79)}
ORIGINS = (59, 63, 67, 71, 75)
PILOT_FREEZE_SIGNATURE = "cdac164df87d4f559e36795066cc0c3a91034e5917b0b1fd72cebd1051dae08b"
CONT_FREEZE_SIGNATURE = "93a2f388673e5cbb2fde60364fd0121918695ecb908c9b2c9c8b91e239de54e6"
MIN_GROUP_ROWS = 200
MIN_CLASS_ROWS = 30
THRESHOLD = 0.50
SHRINKAGE = 4.0
CORRECTION_LIMIT = 0.05
QUERY_TEXT_BLOCK = 2048

FIT_CALLS = 0


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {_rel(path)}")
    return value


def _array_sha256(values: Any, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype)
    if array.dtype.kind in "OUS":
        digest = hashlib.sha256()
        for value in array.reshape(-1):
            payload = str(value).encode("utf-8", errors="surrogatepass")
            digest.update(len(payload).to_bytes(8, "little", signed=False))
            digest.update(payload)
        return digest.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _ids_sha(values: Iterable[Any]) -> str:
    ids = [str(value) for value in values]
    payload = json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_id(value: Any) -> str:
    """Strip only serialization type tags; preserve the underlying official ID."""
    text = str(value)
    if text.startswith("str_:") or text.startswith("str:"):
        return text.split(":", 1)[1]
    return text


@contextmanager
def forbid_fit_calls():
    """Fail closed if this audit accidentally invokes any supported fit path."""
    global FIT_CALLS
    blocked = []

    def reject(*_args: Any, **_kwargs: Any) -> Any:
        global FIT_CALLS
        FIT_CALLS += 1
        raise RuntimeError("A02 reliability audit is hard zero-fit; estimator fit was blocked")

    for cls, method_name in ((ResidualNeighborPredictor, "fit"),
                             (TfidfVectorizer, "fit"),
                             (TfidfVectorizer, "fit_transform")):
        original = getattr(cls, method_name)
        blocked.append((cls, method_name, original))
        setattr(cls, method_name, reject)
    try:
        yield
    finally:
        for cls, method_name, original in reversed(blocked):
            setattr(cls, method_name, original)


def verify_freezes_and_inputs() -> dict[str, Any]:
    pilot_protocol = _json(ROOT / PILOT_PROTOCOL_REL)
    pilot_freeze = _json(ROOT / PILOT_FREEZE_REL)
    cont_protocol = _json(ROOT / CONT_PROTOCOL_REL)
    cont_freeze = _json(ROOT / CONT_FREEZE_REL)
    checks: dict[str, bool] = {}
    checks["pilot_protocol_hash"] = canonical_sha256(pilot_protocol) == pilot_freeze["protocol_sha256"]
    checks["continuation_protocol_hash"] = canonical_sha256(cont_protocol) == cont_freeze["protocol_sha256"]
    checks["pilot_freeze_signature"] = canonical_sha256(pilot_freeze["identity"]) == pilot_freeze["freeze_signature"]
    checks["continuation_freeze_signature"] = canonical_sha256(cont_freeze["identity"]) == cont_freeze["freeze_signature"]
    if pilot_freeze.get("freeze_signature") != PILOT_FREEZE_SIGNATURE:
        raise RuntimeError("The preserved A02 pilot freeze signature changed")
    if cont_freeze.get("freeze_signature") != CONT_FREEZE_SIGNATURE:
        raise RuntimeError("The preserved A02 continuation freeze signature changed")
    verified_paths: dict[str, str] = {}
    for freeze in (pilot_freeze, cont_freeze):
        for relative, expected in freeze.get("identity", {}).get("input_sha256", {}).items():
            path = ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Frozen A02 input hash mismatch: {relative}")
            verified_paths[relative] = expected
        for relative, expected in freeze.get("source_sha256", {}).items():
            path = ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Frozen A02 source hash mismatch: {relative}")
    plan_path = ROOT / PLAN_REL
    if not plan_path.is_file():
        raise RuntimeError("Frozen analysis plan is missing")
    train_path = ROOT / TRAIN_REL
    train_hash = sha256_file(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Classification Train hash differs from frozen value")
    checks["official_train_hash"] = True
    checks["all_frozen_parent_inputs"] = True
    checks["no_test_sample_or_kaggle_paths_in_scope"] = True
    return {
        "checks": checks,
        "pilot_freeze_signature": pilot_freeze["freeze_signature"],
        "continuation_freeze_signature": cont_freeze["freeze_signature"],
        "pilot_protocol": pilot_protocol,
        "continuation_protocol": cont_protocol,
        "input_hashes_verified": verified_paths,
        "train_sha256": train_hash,
        "analysis_plan_sha256": sha256_file(plan_path),
        "freeze_objects": {"pilot": pilot_freeze, "continuation": cont_freeze},
    }


def load_train() -> pd.DataFrame:
    path = ROOT / TRAIN_REL
    frame = pd.read_csv(path, usecols=["record_id", "cycle_code", "artifact_signal", "signal_stream"],
                        dtype={"record_id": str})
    if (len(frame) != 50_608 or frame["record_id"].isna().any()
            or frame["record_id"].duplicated().any()):
        raise RuntimeError("Official Train row identity/count invalid")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int16)
    frame["artifact_signal"] = pd.to_numeric(frame["artifact_signal"], errors="raise").astype(np.int8)
    if set(frame["artifact_signal"].unique()) != {0, 1}:
        raise RuntimeError("Official Train target is not binary")
    return frame


def verify_row_join(train: pd.DataFrame, rows: Any, ids: Any, cycles: Any) -> np.ndarray:
    """Join labels by exact source row + ID + cycle; never infer time from row order."""
    row_array = np.asarray(rows)
    id_array = np.asarray(ids).astype(str)
    cycle_array = np.asarray(cycles, dtype=np.int64)
    if (row_array.ndim != 1 or row_array.dtype.kind not in "iu" or len(row_array) == 0
            or id_array.shape != row_array.shape or cycle_array.shape != row_array.shape):
        raise ValueError("row/ID/cycle join arrays must be aligned nonempty vectors")
    row_array = row_array.astype(np.int64, copy=False)
    if (np.any(row_array < 0) or np.any(row_array >= len(train))
            or len(np.unique(row_array)) != len(row_array) or np.any(np.diff(row_array) <= 0)):
        raise ValueError("source row positions must be in-bounds, unique, and strictly increasing")
    if not np.array_equal(id_array, train["record_id"].to_numpy()[row_array]):
        raise ValueError("record_id mismatch at exact Train row position")
    expected_cycle = train["cycle_code"].to_numpy(dtype=np.int64)[row_array]
    if not np.array_equal(cycle_array, expected_cycle):
        raise ValueError("cycle_code mismatch at exact Train row position")
    return train["artifact_signal"].to_numpy(dtype=np.int8)[row_array].copy()


def load_c16_caches(train: pd.DataFrame, freeze_state: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    caches: dict[int, dict[str, np.ndarray]] = {}
    frozen_inputs = {}
    for freeze in freeze_state["freeze_objects"].values():
        frozen_inputs.update(freeze.get("identity", {}).get("input_sha256", {}))
    for cutoff, relative in CACHE_PATHS.items():
        path = ROOT / relative
        side_path = path.with_suffix(".json")
        side = _json(side_path)
        if not path.is_file() or not side_path.is_file() or sha256_file(path) != side.get("npz_sha256"):
            raise RuntimeError(f"C16 cache/sidecar hash mismatch at cutoff {cutoff}")
        if side.get("identity_sha256") != canonical_sha256(side.get("identity")):
            raise RuntimeError(f"C16 cache identity signature failed at cutoff {cutoff}")
        if side.get("integrity_sha256") != canonical_sha256({k: v for k, v in side.items() if k != "integrity_sha256"}):
            raise RuntimeError(f"C16 cache sidecar integrity failed at cutoff {cutoff}")
        expected_npz = frozen_inputs.get(relative)
        expected_side = frozen_inputs.get(side_path.relative_to(ROOT).as_posix())
        if expected_npz and expected_npz != sha256_file(path):
            raise RuntimeError(f"C16 cache NPZ differs from frozen input at cutoff {cutoff}")
        if expected_side and expected_side != sha256_file(side_path):
            raise RuntimeError(f"C16 cache sidecar differs from frozen input at cutoff {cutoff}")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: np.asarray(archive[key]).copy() for key in (
                "score_rows", "score_record_ids", "baseline_prediction",
                "control__baseline_sparse_raw", "control__c2b_probability",
                "control__cb_probability", "control__stream_missing", "control__control_rows",
                "audit__stream_missing")}
        rows = data["score_rows"].astype(np.int64)
        ids = data["score_record_ids"].astype(str)
        cycles = train["cycle_code"].to_numpy(dtype=np.int16)[rows]
        missing = train["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
        low, high = CACHE_WINDOWS[cutoff]
        expected_rows = np.flatnonzero(train["cycle_code"].between(low, high).to_numpy())
        cycle_counts_match = all(np.count_nonzero(cycles == c) == np.count_nonzero(train["cycle_code"].to_numpy() == c)
                                 for c in range(low, high + 1))
        if (not np.array_equal(rows, expected_rows) or len(np.unique(rows)) != len(rows)
                or np.any(np.diff(rows) <= 0) or not np.array_equal(data["control__control_rows"], rows)
                or not np.array_equal(np.unique(cycles), np.arange(low, high + 1, dtype=np.int16))
                or not cycle_counts_match
                or not np.array_equal(data["control__stream_missing"], missing)
                or not np.array_equal(data["audit__stream_missing"], missing)):
            raise RuntimeError(f"C16 cache query rows/IDs/cycles/mask fail at cutoff {cutoff}")
        verify_row_join(train, rows, ids, cycles)
        p_base = data["baseline_prediction"].astype(np.float64)
        reconstruction = np.where(missing, data["control__cb_probability"],
                                  0.75 * data["control__c2b_probability"]
                                  + 0.25 * data["control__baseline_sparse_raw"])
        if not np.array_equal(p_base, reconstruction):
            raise RuntimeError(f"C16 baseline route is not bit-exact Sparse25 at cutoff {cutoff}")
        if np.any(~np.isfinite(p_base)) or np.any((p_base < 0) | (p_base > 1)):
            raise RuntimeError(f"C16 baseline probabilities invalid at cutoff {cutoff}")
        data.update({"score_rows": rows, "score_record_ids": ids, "score_cycles": cycles,
                     "baseline_prediction": p_base, "missing_mask": missing})
        caches[cutoff] = data
    return caches


def _bank_blocks(origin: int, pilot_protocol: dict[str, Any],
                 cont_protocol: dict[str, Any]) -> list[tuple[int, int, int]]:
    if origin in (59, 63, 71):
        return [tuple(map(int, item)) for item in cont_protocol["oof_bank_blocks"][str(origin)]]
    return [tuple(map(int, item)) for item in pilot_protocol["bank"][f"E{origin}"]["source_blocks"]]


def expected_bank(origin: int, caches: dict[int, dict[str, np.ndarray]], train: pd.DataFrame,
                  pilot_protocol: dict[str, Any], cont_protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {k: [] for k in ("rows", "ids", "cycles", "baseline", "oof_origin")}
    for cutoff, low, high in _bank_blocks(origin, pilot_protocol, cont_protocol):
        cache = caches[cutoff]
        mask = ((cache["score_cycles"] >= low) & (cache["score_cycles"] <= high)
                & ~cache["missing_mask"])
        pieces["rows"].append(cache["score_rows"][mask])
        pieces["ids"].append(cache["score_record_ids"][mask])
        pieces["cycles"].append(cache["score_cycles"][mask])
        pieces["baseline"].append(cache["baseline_prediction"][mask])
        pieces["oof_origin"].append(np.full(int(mask.sum()), cutoff, dtype=np.int16))
    joined = {key: np.concatenate(value) for key, value in pieces.items()}
    order = np.argsort(joined["rows"], kind="stable")
    joined = {key: value[order] for key, value in joined.items()}
    if len(np.unique(joined["rows"])) != len(joined["rows"]):
        raise RuntimeError(f"OOF bank rows duplicate at origin {origin}")
    if (not np.array_equal(joined["ids"], train["record_id"].to_numpy()[joined["rows"]])
            or not np.array_equal(joined["cycles"], train["cycle_code"].to_numpy(dtype=np.int16)[joined["rows"]])
            or np.any(joined["oof_origin"] >= joined["cycles"])
            or np.any(joined["cycles"] > origin)):
        raise RuntimeError(f"OOF bank row/ID/cycle provenance invalid at origin {origin}")
    return joined


def _safe_pred_paths(origin: int, arm: str) -> Path:
    root_name = "optimization_v3_a02_continuation_v1" if origin in (59, 63, 71) else "optimization_v3_a02_neighbors_v1"
    return ROOT / "experiments/runs" / root_name / f"E{origin}" / arm


def load_prediction(path: Path, origin: int, arm: str, train: pd.DataFrame,
                    caches: dict[int, dict[str, np.ndarray]], freeze_state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any]]:
    side_path, receipt_path, state_path = path / "predictions.json", path / "fit_receipt.json", path / "fit_state.json"
    side, receipt, state = _json(side_path), _json(receipt_path), _json(state_path)
    npz_path = path / "predictions.npz"
    npz_hash = sha256_file(npz_path)
    if side.get("npz_sha256") != npz_hash or side.get("origin_cycle") != origin or side.get("residual_arm") != arm:
        raise RuntimeError(f"Saved A02 prediction identity/hash mismatch E{origin}/{arm}")
    if receipt.get("model_sha256") != sha256_file(path / "model.joblib"):
        raise RuntimeError(f"A02 model hash mismatch E{origin}/{arm}")
    if side.get("fit_receipt_sha256") != receipt.get("receipt_sha256"):
        raise RuntimeError(f"A02 prediction receipt link mismatch E{origin}/{arm}")
    expected_freeze = CONT_FREEZE_SIGNATURE if origin in (59, 63, 71) else PILOT_FREEZE_SIGNATURE
    if (side.get("freeze_signature") != expected_freeze or receipt.get("freeze_signature") != expected_freeze
            or side.get("query_labels_passed_to_prediction") is not False
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False):
        raise RuntimeError(f"A02 data-boundary/freeze attestation mismatch E{origin}/{arm}")
    if side.get("test_data_used", False) is not False:
        raise RuntimeError(f"A02 sidecar attests Test use E{origin}/{arm}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
                "labels", "missing_mask", "neighbor_count", "neighbor_source_rows", "neighbor_ids",
                "neighbor_similarities", "correction", "correction_unclipped"}
    if not required.issubset(arrays):
        raise RuntimeError(f"A02 NPZ missing required fields E{origin}/{arm}")
    rows = arrays["score_rows"].astype(np.int64)
    window_low, window_high = CACHE_WINDOWS[origin]
    expected_rows = np.flatnonzero(((train["cycle_code"].to_numpy(dtype=np.int64) >= window_low)
                                    & (train["cycle_code"].to_numpy(dtype=np.int64) <= window_high)))
    if (not np.array_equal(rows, expected_rows) or len(np.unique(rows)) != len(rows)
            or np.any(np.diff(rows) <= 0)):
        raise RuntimeError(f"A02 query rows/IDs/cycles mismatch E{origin}/{arm}")
    verify_row_join(train, rows, arrays["score_record_ids"], arrays["score_cycles"])
    miss = train["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
    if not np.array_equal(arrays["missing_mask"].astype(bool), miss):
        raise RuntimeError(f"A02 missing-stream mask mismatch E{origin}/{arm}")
    route = caches[origin]
    base_by_row = dict(zip(route["score_rows"].tolist(), route["baseline_prediction"].tolist()))
    expected_base = np.asarray([base_by_row[int(row)] for row in rows], dtype=np.float64)
    if not np.array_equal(arrays["baseline_prediction"].astype(np.float64), expected_base):
        raise RuntimeError(f"A02 baseline differs from frozen C16 Sparse25 bridge E{origin}/{arm}")
    if side.get("score_rows") != len(rows) or side.get("score_row_positions_sha256") != _array_sha256(rows, "<i8"):
        raise RuntimeError(f"A02 sidecar score-row integrity mismatch E{origin}/{arm}")
    if side.get("score_record_ids_sha256") != _ids_sha(arrays["score_record_ids"]):
        raise RuntimeError(f"A02 sidecar score-ID integrity mismatch E{origin}/{arm}")
    if not np.array_equal(arrays["labels"].astype(np.int8), (arrays["prediction"] >= THRESHOLD).astype(np.int8)):
        raise RuntimeError(f"A02 NPZ predicted-label field does not match saved probabilities E{origin}/{arm}")
    return arrays, side, receipt, state


def load_and_verify_model(path: Path, origin: int, arm: str, receipt: dict[str, Any],
                          state: dict[str, Any], bank: dict[str, np.ndarray],
                          train: pd.DataFrame, caches: dict[int, dict[str, np.ndarray]]) -> Any:
    model = joblib.load(path / "model.joblib")
    if (not isinstance(model, ResidualNeighborPredictor) or not model._fitted
            or model.origin_cycle != origin or model.residual_arm != arm):
        raise RuntimeError(f"Persisted model identity or fitted-state invalid E{origin}/{arm}")
    if model._state != state:
        raise RuntimeError(f"Persisted model fit state differs from sidecar E{origin}/{arm}")
    if canonical_sha256({k: v for k, v in state.items() if k != "state_sha256"}) != state.get("state_sha256"):
        raise RuntimeError(f"Persisted state self-digest invalid E{origin}/{arm}")
    expected_rows = bank["rows"]
    expected_ids = bank["ids"]
    if (not np.array_equal(model.bank_row_positions, expected_rows)
            or [normalize_id(v) for v in model.bank_ids] != expected_ids.tolist()
            or not np.array_equal(model.bank_cycles, bank["cycles"])
            or not np.array_equal(model.bank_oof_probabilities, bank["baseline"])):
        raise RuntimeError(f"Persisted model OOF bank differs from verified C16 bank E{origin}/{arm}")
    expected_residual = train["artifact_signal"].to_numpy(dtype=np.int8)[bank["rows"]].astype(np.float64) - bank["baseline"]
    if arm == "observed":
        expected_assignment = np.arange(len(expected_residual), dtype=np.int64)
        assigned = expected_residual
    else:
        assigned, expected_assignment, _ = permute_residuals_within_cycle_probability_decile(
            expected_residual, bank["cycles"], bank["baseline"], bank["rows"], seed=42)
    if (not np.array_equal(model.residual_assignment, expected_assignment)
            or not np.array_equal(model.bank_residuals, assigned)):
        raise RuntimeError(f"Persisted residual arm is inconsistent with frozen OOF assignment E{origin}/{arm}")
    if (not np.allclose(model.bank_residuals, assigned, rtol=0, atol=0)
            or receipt.get("model_sha256") != sha256_file(path / "model.joblib")
            or receipt.get("origin_cycle") != origin or receipt.get("residual_arm") != arm):
        raise RuntimeError(f"Persisted model/receipt audit failed E{origin}/{arm}")
    if model._state.get("bank_rows") != len(bank["rows"]):
        raise RuntimeError(f"Persisted OOF bank row count mismatch E{origin}/{arm}")
    return model


def zero_vector_flags(model: Any, train: pd.DataFrame, rows: np.ndarray,
                      missing: np.ndarray) -> np.ndarray:
    result = np.zeros(len(rows), dtype=bool)
    texts = train["signal_stream"].to_numpy(dtype=object)[rows]
    for start in range(0, len(rows), QUERY_TEXT_BLOCK):
        stop = min(start + QUERY_TEXT_BLOCK, len(rows))
        matrix, transformed_missing = model.text_space.transform(texts[start:stop].tolist())
        if not np.array_equal(transformed_missing, missing[start:stop]):
            raise RuntimeError("Stored text-space missing normalization differs from saved output")
        zero = np.asarray(matrix.getnnz(axis=1) == 0, dtype=bool)
        result[start:stop] = zero & ~missing[start:stop]
    return result


def reconstruct_descriptors(arr: dict[str, np.ndarray], model: Any, train: pd.DataFrame,
                             origin: int) -> dict[str, np.ndarray]:
    rows = arr["score_rows"].astype(np.int64)
    cycles = arr["score_cycles"].astype(np.int64)
    missing = arr["missing_mask"].astype(bool)
    counts = arr["neighbor_count"].astype(np.int64)
    positions = arr["neighbor_source_rows"].astype(np.int64)
    neighbor_ids = arr["neighbor_ids"].astype(str)
    sims = arr["neighbor_similarities"].astype(np.float64)
    base = arr["baseline_prediction"].astype(np.float64)
    pred = arr["prediction"].astype(np.float64)
    raw_saved = arr["correction_unclipped"].astype(np.float64)
    clipped_saved = arr["correction"].astype(np.float64)
    bank_index = {int(row): i for i, row in enumerate(model.bank_row_positions)}
    n = len(rows)
    mass = np.zeros(n); ess = np.zeros(n); max_cos = np.zeros(n); shrink = np.zeros(n)
    resid_mean = np.full(n, np.nan); resid_sd = np.full(n, np.nan)
    origin_age = np.full(n, np.nan); query_age = np.full(n, np.nan)
    last4_mass = np.zeros(n); last4_fraction = np.full(n, np.nan)
    represented_cycles = np.zeros(n, dtype=np.int16); cycle_concentration = np.full(n, np.nan)
    sign_consistency = np.full(n, np.nan); tie = np.zeros(n, dtype=bool); boundary_tie = np.zeros(n, dtype=bool)
    single_effective = np.zeros(n, dtype=bool); raw_rebuilt = np.zeros(n)
    for i, count in enumerate(counts):
        if count < 0 or count > positions.shape[1]:
            raise RuntimeError(f"Invalid saved neighbor count at origin {origin}")
        rows_i, ids_i, sims_i = positions[i, :count], neighbor_ids[i, :count], sims[i, :count]
        if (np.any(positions[i, count:] != -1) or np.any(neighbor_ids[i, count:] != "")
                or np.any(sims[i, count:] != 0) or np.any(sims_i <= 0)):
            raise RuntimeError(f"Neighbor padding or positive-weight contract failed at origin {origin}")
        if count == 0:
            if not missing[i] and (raw_saved[i] != 0 or clipped_saved[i] != 0):
                raise RuntimeError(f"Zero-neighbor correction is not zero at origin {origin}")
            if missing[i] and (raw_saved[i] != 0 or clipped_saved[i] != 0 or pred[i] != base[i]):
                raise RuntimeError(f"Missing-stream exact fallback failed at origin {origin}")
            continue
        if (len(np.unique(rows_i)) != count or np.any(rows_i == rows[i])
                or np.any(cycles[i] <= model.bank_cycles[[bank_index[int(r)] for r in rows_i]])):
            raise RuntimeError(f"A02 neighbor row/self/future-cycle check failed at origin {origin}")
        idx = np.asarray([bank_index[int(row)] for row in rows_i], dtype=np.int64)
        if [normalize_id(v) for v in ids_i] != [normalize_id(model.bank_ids[j]) for j in idx]:
            raise RuntimeError(f"Neighbor ID does not match persisted bank row at origin {origin}")
        if (np.any(np.diff(sims_i) > 1e-15)
                or any((sims_i[j] == sims_i[j + 1] and rows_i[j] > rows_i[j + 1])
                       for j in range(count - 1))):
            raise RuntimeError(f"Neighbor sort/tie order invalid at origin {origin}")
        tie[i] = len(np.unique(sims_i)) < count
        boundary_tie[i] = count == 32 and np.count_nonzero(sims_i == sims_i[-1]) > 1
        weights = sims_i * sims_i
        total = float(weights.sum())
        if total <= 0:
            raise RuntimeError("A positive-neighbor row has zero total weight")
        mass[i] = total
        ess[i] = total * total / float(np.dot(weights, weights))
        max_cos[i] = float(sims_i.max())
        shrink[i] = total / (SHRINKAGE + total)
        residuals = model.bank_residuals[idx]
        resid_mean[i] = float(np.dot(weights, residuals) / total)
        resid_sd[i] = math.sqrt(max(0.0, float(np.dot(weights, (residuals - resid_mean[i]) ** 2) / total)))
        raw_rebuilt[i] = float(np.dot(weights, residuals) / (SHRINKAGE + total))
        bcycles = model.bank_cycles[idx]
        origin_ages = origin - bcycles
        query_ages = cycles[i] - bcycles
        origin_age[i] = float(np.dot(weights, origin_ages) / total)
        query_age[i] = float(np.dot(weights, query_ages) / total)
        last4 = bcycles >= origin - 3
        last4_mass[i] = float(weights[last4].sum())
        last4_fraction[i] = last4_mass[i] / total
        represented_cycles[i] = int(len(np.unique(bcycles)))
        cycle_masses = [float(weights[bcycles == c].sum()) for c in np.unique(bcycles)]
        cycle_concentration[i] = max(cycle_masses) / total
        pos_mass = float(weights[residuals > 0].sum())
        neg_mass = float(weights[residuals < 0].sum())
        zero_mass = float(weights[residuals == 0].sum())
        sign_consistency[i] = max(pos_mass, neg_mass, zero_mass) / total
        single_effective[i] = ess[i] <= 1.0 + 1e-12
    if not np.allclose(raw_saved, raw_rebuilt, rtol=0, atol=1e-15):
        raise RuntimeError(f"Saved raw correction does not reconstruct from bank residuals at origin {origin}")
    expected_clipped = np.clip(raw_rebuilt, -CORRECTION_LIMIT, CORRECTION_LIMIT)
    expected_clipped[missing] = 0.0
    if not np.array_equal(clipped_saved, expected_clipped):
        raise RuntimeError(f"Saved clipped correction does not reconstruct at origin {origin}")
    expected_prediction = np.clip(base + clipped_saved, 0.0, 1.0)
    expected_prediction[missing] = base[missing]
    if not np.array_equal(expected_prediction, pred):
        raise RuntimeError(f"Saved A02 probability does not reconstruct at origin {origin}")
    zero_text = zero_vector_flags(model, train, rows, missing)
    if np.any(zero_text & (counts > 0)):
        raise RuntimeError("A zero-vector query has positive saved neighbors")
    return {
        "max_cosine": max_cos, "similarity_mass": mass, "ess": ess,
        "positive_neighbor_count": counts.astype(np.int16), "shrink_factor": shrink,
        "weighted_residual_mean": resid_mean, "weighted_residual_sd": resid_sd,
        "abs_raw_correction": np.abs(raw_saved), "cap_hit_or_boundary": np.abs(raw_saved) >= CORRECTION_LIMIT,
        "strict_clip_hit": np.abs(raw_saved) > CORRECTION_LIMIT,
        "baseline_probability": base, "mean_origin_age": origin_age, "mean_query_age": query_age,
        "last4_cycle_mass": last4_mass, "last4_cycle_mass_fraction": last4_fraction,
        "represented_cycle_count": represented_cycles, "max_cycle_weight_concentration": cycle_concentration,
        "weighted_residual_sign_consistency": sign_consistency,
        "zero_tfidf_vector": zero_text, "no_positive_weights": counts == 0,
        "single_effective_neighbor": single_effective, "selected_similarity_tie": tie,
        "saved_top32_boundary_tie": boundary_tie,
    }


def _load_budget() -> dict[str, Any]:
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    started_ids = [e.get("fit_id") for e in starts]
    if len(started_ids) != len(set(started_ids)):
        raise RuntimeError("Research fit ledger contains duplicate fit starts")
    return {"ledger_path": _rel(LEDGER_PATH), "ledger_sha256": sha256_file(LEDGER_PATH),
            "event_rows": len(events), "fit_starts": len(starts), "fit_saved": len(saved),
            "fit_failed": len(failed), "cap": 66,
            "remaining_capacity_reported": max(0, 66 - len(starts)),
            "fit_ids_unique": True}


def build_origin_frame(origin: int, train: pd.DataFrame,
                       caches: dict[int, dict[str, np.ndarray]], freeze_state: dict[str, Any]) -> dict[str, np.ndarray]:
    loaded: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    bank = expected_bank(origin, caches, train,
                         freeze_state["pilot_protocol"], freeze_state["continuation_protocol"])
    feature_arrays = None
    for arm in ("observed", "shuffled"):
        path = _safe_pred_paths(origin, arm)
        arr, side, receipt, state = load_prediction(path, origin, arm, train, caches, freeze_state)
        model = load_and_verify_model(path, origin, arm, receipt, state, bank, train, caches)
        if arm == "observed":
            feature_arrays = reconstruct_descriptors(arr, model, train, origin)
        loaded[arm] = (arr, side)
        del model
        gc.collect()
    obs, shuffled = loaded["observed"][0], loaded["shuffled"][0]
    for key in ("score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "missing_mask",
                "neighbor_count", "neighbor_source_rows", "neighbor_ids", "neighbor_similarities"):
        if not np.array_equal(obs[key], shuffled[key]):
            raise RuntimeError(f"Observed and shuffled geometry/route differ at E{origin}/{key}")
    rows = obs["score_rows"].astype(np.int64)
    truth = verify_row_join(train, rows, obs["score_record_ids"], obs["score_cycles"])
    return {
        "origin": np.full(len(rows), origin, dtype=np.int16),
        "rows": rows,
        "cycles": obs["score_cycles"].astype(np.int16),
        "y": truth,
        "baseline": obs["baseline_prediction"].astype(np.float64),
        "observed": obs["prediction"].astype(np.float64),
        "shuffled": shuffled["prediction"].astype(np.float64),
        "missing": obs["missing_mask"].astype(bool),
        **(feature_arrays or {}),
    }


FEATURE_KEYS = (
    "max_cosine", "similarity_mass", "ess", "positive_neighbor_count", "shrink_factor",
    "weighted_residual_mean", "weighted_residual_sd", "abs_raw_correction",
    "cap_hit_or_boundary", "strict_clip_hit", "baseline_probability", "mean_origin_age",
    "mean_query_age", "last4_cycle_mass", "last4_cycle_mass_fraction",
    "represented_cycle_count", "max_cycle_weight_concentration",
    "weighted_residual_sign_consistency", "zero_tfidf_vector", "no_positive_weights",
    "single_effective_neighbor", "selected_similarity_tie", "saved_top32_boundary_tie",
)
FRAME_KEYS = ("origin", "rows", "cycles", "y", "baseline", "observed", "shuffled", "missing", *FEATURE_KEYS)


def _missing_feature_arrays(n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in FEATURE_KEYS:
        if key in {"cap_hit_or_boundary", "strict_clip_hit", "zero_tfidf_vector", "no_positive_weights",
                   "single_effective_neighbor", "selected_similarity_tie", "saved_top32_boundary_tie"}:
            out[key] = np.zeros(n, dtype=bool)
        elif key == "positive_neighbor_count" or key == "represented_cycle_count":
            out[key] = np.zeros(n, dtype=np.int16)
        else:
            out[key] = np.full(n, np.nan, dtype=np.float64)
    return out


def build_w1_frame(train: pd.DataFrame, caches: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    cache = caches[55]
    select = (cache["score_cycles"] >= 56) & (cache["score_cycles"] <= 59)
    rows = cache["score_rows"][select]
    base = cache["baseline_prediction"][select]
    missing = cache["missing_mask"][select]
    return {
        "origin": np.full(len(rows), -1, dtype=np.int16),
        "rows": rows,
        "cycles": cache["score_cycles"][select].astype(np.int16),
        "y": train["artifact_signal"].to_numpy(dtype=np.int8)[rows],
        "baseline": base.copy(), "observed": base.copy(), "shuffled": base.copy(), "missing": missing.copy(),
        **_missing_feature_arrays(len(rows)),
    }


def slice_frame(frame: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in frame.items()}


def concat_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not frames:
        return {key: np.array([], dtype=np.float64) for key in FRAME_KEYS}
    return {key: np.concatenate([frame[key] for frame in frames]) for key in FRAME_KEYS}


def make_view(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    view = concat_frames(frames)
    rows = view["rows"].astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise RuntimeError("Evaluation view contains duplicate Train row positions")
    order = np.argsort(rows, kind="stable")
    return {key: value[order] for key, value in view.items()}


def build_views(train: pd.DataFrame, origin_frames: dict[int, dict[str, np.ndarray]],
                caches: dict[int, dict[str, np.ndarray]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    w1 = build_w1_frame(train, caches)
    first4 = {origin: slice_frame(frame, (frame["cycles"] >= lo) & (frame["cycles"] <= hi))
              for origin, frame in origin_frames.items()
              for lo, hi in [ORIGIN_FIRST4[origin]]}
    views = {
        "W1": w1,
        "W2": first4[59], "W3": first4[63], "W4": first4[67],
        "W5": first4[71], "W6": first4[75],
        "new_origin_aggregate": make_view([first4[59], first4[63], first4[71]]),
        "full_six_primary": make_view([w1, first4[59], first4[63], first4[67], first4[71], first4[75]]),
        "late": make_view([first4[71], first4[75]]),
        "inherited_H12_L": slice_frame(origin_frames[67],
                                         (origin_frames[67]["cycles"] >= 68)
                                         & (origin_frames[67]["cycles"] <= 79)),
        "inherited_pilot_primary_W4_W6": make_view([first4[67], first4[75]]),
    }
    views.update({f"E{origin}_all_available": frame for origin, frame in origin_frames.items()})
    expected_counts = {"W1": 4114, "W2": 6493, "W3": 7078, "W4": 4950,
                       "W5": 7462, "W6": 8491, "new_origin_aggregate": 21033,
                       "full_six_primary": 38588, "late": 15953,
                       "inherited_H12_L": 20903, "inherited_pilot_primary_W4_W6": 13441}
    for key, expected in expected_counts.items():
        if len(views[key]["rows"]) != expected:
            raise RuntimeError(f"Evaluation view {key} has {len(views[key]['rows'])} rows; expected {expected}")
    overlap = np.intersect1d(views["full_six_primary"]["rows"], views["inherited_H12_L"]["rows"])
    if len(overlap) != len(views["inherited_H12_L"]["rows"]):
        raise RuntimeError("H12-L query overlap differs from its declared same-cycle primary rows")
    provenance = {
        "W1_exact_sparse25_fallback_rows": len(w1["rows"]),
        "new_origin_rows_unique": len(views["new_origin_aggregate"]["rows"]),
        "primary_rows_unique": len(views["full_six_primary"]["rows"]),
        "late_rows_unique": len(views["late"]["rows"]),
        "H12_L_rows": len(views["inherited_H12_L"]["rows"]),
        "H12_L_rows_overlapping_primary": len(overlap),
        "primary_h12_union_unique_rows": len(np.union1d(views["full_six_primary"]["rows"],
                                                          views["inherited_H12_L"]["rows"])),
        "note": "H12-L uses the same query rows as primary W4-W6 under the older origin-67 prediction; it is a separate overlapping diagnostic, not independent evidence.",
    }
    return views, provenance


def _confusion(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.int8)
    probability = np.asarray(p, dtype=np.float64)
    predicted = probability >= THRESHOLD
    tn = int(np.sum((truth == 0) & ~predicted))
    fp = int(np.sum((truth == 0) & predicted))
    fn = int(np.sum((truth == 1) & ~predicted))
    tp = int(np.sum((truth == 1) & predicted))
    f1_zero = (2 * tn / (2 * tn + fp + fn)) if 2 * tn + fp + fn else 0.0
    f1_one = (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {"n": int(len(truth)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "macro_f1": float((f1_zero + f1_one) / 2.0),
            "positive_recall": float(tp / (tp + fn)) if tp + fn else None,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "classification_error": float((fp + fn) / len(truth)) if len(truth) else None}


def _row_logloss(y: np.ndarray, p: np.ndarray, epsilon: float) -> np.ndarray:
    probability = np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon)
    truth = np.asarray(y, dtype=np.int8)
    return -(truth * np.log(probability) + (1 - truth) * np.log1p(-probability))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.int8)
    probability = np.asarray(p, dtype=np.float64)
    if not len(truth):
        return {"n": 0, "roc_auc": None, "average_precision": None, "brier": None,
                "logloss_e15": None, "logloss_native_float64_epsilon": None,
                "exact_endpoint_rows": 0, "wrong_endpoint_rows": 0}
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Metric probabilities must be finite in [0,1]")
    auc = float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None
    ap = float(average_precision_score(truth, probability)) if len(np.unique(truth)) == 2 else None
    endpoint = (probability == 0.0) | (probability == 1.0)
    wrong = ((probability == 0.0) & (truth == 1)) | ((probability == 1.0) & (truth == 0))
    return {"n": int(len(truth)), "roc_auc": auc, "average_precision": ap,
            "brier": float(np.mean((probability - truth) ** 2)),
            "logloss_e15": float(np.mean(_row_logloss(truth, probability, 1e-15))),
            "logloss_native_float64_epsilon": float(np.mean(_row_logloss(truth, probability, np.finfo(np.float64).eps))),
            "exact_endpoint_rows": int(endpoint.sum()), "wrong_endpoint_rows": int(wrong.sum()),
            "wrong_p0_y1": int(((probability == 0.0) & (truth == 1)).sum()),
            "wrong_p1_y0": int(((probability == 1.0) & (truth == 0)).sum())}


def additive_diagnostics(frame: dict[str, np.ndarray]) -> dict[str, Any]:
    y = frame["y"].astype(np.int8)
    base, obs, shuf = (frame[k].astype(np.float64) for k in ("baseline", "observed", "shuffled"))
    base_brier, obs_brier, shuf_brier = (np.square(p - y) for p in (base, obs, shuf))
    base_error, obs_error, shuf_error = (np.not_equal(p >= THRESHOLD, y).astype(np.float64)
                                         for p in (base, obs, shuf))
    base_ll15, obs_ll15, shuf_ll15 = (_row_logloss(y, p, 1e-15) for p in (base, obs, shuf))
    eps = np.finfo(np.float64).eps
    base_llnative, obs_llnative, shuf_llnative = (_row_logloss(y, p, eps) for p in (base, obs, shuf))
    wrong_endpoint = ((obs == 0.0) & (y == 1)) | ((obs == 1.0) & (y == 0))
    other = ~wrong_endpoint

    def mean_or_none(values: np.ndarray) -> float | None:
        return float(values.mean()) if len(values) else None

    result = {
        "brier_improvement_vs_sparse25": mean_or_none(base_brier - obs_brier),
        "brier_improvement_vs_shuffle": mean_or_none(shuf_brier - obs_brier),
        "classification_error_improvement_vs_sparse25": mean_or_none(base_error - obs_error),
        "classification_error_improvement_vs_shuffle": mean_or_none(shuf_error - obs_error),
        "logloss_improvement_vs_sparse25_e15": mean_or_none(base_ll15 - obs_ll15),
        "logloss_improvement_vs_shuffle_e15": mean_or_none(shuf_ll15 - obs_ll15),
        "logloss_improvement_vs_sparse25_native_epsilon": mean_or_none(base_llnative - obs_llnative),
        "logloss_improvement_vs_shuffle_native_epsilon": mean_or_none(shuf_llnative - obs_llnative),
        "candidate_exact_endpoint_rows": int(np.isin(obs, [0.0, 1.0]).sum()),
        "candidate_wrong_exact_endpoint_rows": int(wrong_endpoint.sum()),
        "wrong_endpoint_breakdown": {"p0_y1": int(((obs == 0.0) & (y == 1)).sum()),
                                     "p1_y0": int(((obs == 1.0) & (y == 0)).sum())},
    }
    for label, mask in (("wrong_endpoint", wrong_endpoint), ("all_other_rows", other)):
        result[label] = {
            "rows": int(mask.sum()),
            "logloss_improvement_vs_sparse25_e15_contribution_to_view_mean":
                float(np.sum((base_ll15 - obs_ll15)[mask]) / len(y)) if len(y) else None,
            "logloss_improvement_vs_sparse25_native_epsilon_contribution_to_view_mean":
                float(np.sum((base_llnative - obs_llnative)[mask]) / len(y)) if len(y) else None,
        }
    return result


def summarize_view(frame: dict[str, np.ndarray]) -> dict[str, Any]:
    y = frame["y"].astype(np.int8)
    n = len(y)
    present_learned = (~frame["missing"].astype(bool)) & (frame["origin"] >= 0)
    arms = {arm: {**_confusion(y, frame[arm]), **probability_metrics(y, frame[arm])}
            for arm in ("baseline", "observed", "shuffled")}
    cycles = np.unique(frame["cycles"]).astype(int).tolist()
    origins = np.unique(frame["origin"][frame["origin"] >= 0]).astype(int).tolist()
    return {"rows": n, "unique_train_rows": len(np.unique(frame["rows"])),
            "cycles": cycles, "origin_sources": origins,
            "stream_present_rows": int((~frame["missing"]).sum()),
            "stream_missing_rows": int(frame["missing"].sum()),
            "learned_present_rows": int(present_learned.sum()),
            "arms": arms, "paired_additive_diagnostics": additive_diagnostics(frame)}


def _fixed_bins(values: np.ndarray, edges: list[float], labels: list[str],
                missing_label: str = "NO_NEIGHBOR") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), missing_label, dtype=object)
    ok = np.isfinite(values)
    if ok.any():
        idx = np.searchsorted(np.asarray(edges[1:-1], dtype=np.float64), values[ok], side="right")
        idx = np.clip(idx, 0, len(labels) - 1)
        result[ok] = np.asarray(labels, dtype=object)[idx]
    return result


SIM_LABELS = ["[0,.2)", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]"]
ESS_LABELS = ["[0,4)", "[4,8)", "[8,16)", "[16,32)", "[32,+inf)"]
PROB_LABELS = [f"[{i/10:.1f},{(i+1)/10:.1f}{']' if i==9 else ')'}" for i in range(10)]
AGE_LABELS = ["[0,4)", "[4,8)", "[8,16)", "[16,32)", "[32,+inf)"]
SIGN_LABELS = ["[0,.5)", "[.5,.6)", "[.6,.7)", "[.7,.8)", "[.8,.9)", "[.9,1]"]
SIM_BROAD_LABELS = ["[0,.4)", "[.4,.6)", "[.6,1]"]


def _metric_record(y: np.ndarray, probability: np.ndarray, supported: bool) -> dict[str, Any]:
    confusion = _confusion(y, probability)
    probabilities = probability_metrics(y, probability)
    if not supported:
        confusion["macro_f1"] = None
        confusion["positive_recall"] = None
        confusion["precision"] = None
    return {**confusion, "brier": probabilities["brier"],
            "logloss_e15": probabilities["logloss_e15"],
            "logloss_native_float64_epsilon": probabilities["logloss_native_float64_epsilon"],
            "endpoint_rows": probabilities["exact_endpoint_rows"],
            "wrong_endpoint_rows": probabilities["wrong_endpoint_rows"]}


def _support_reason(n: int, positives: int, negatives: int) -> tuple[bool, str]:
    reasons = []
    if n == 0:
        return False, "EMPTY"
    if n < MIN_GROUP_ROWS:
        reasons.append("n_below_200")
    if positives < MIN_CLASS_ROWS:
        reasons.append("positive_below_30")
    if negatives < MIN_CLASS_ROWS:
        reasons.append("negative_below_30")
    return not reasons, "SUPPORTED" if not reasons else ";".join(reasons)


def _group_row(view_name: str, frame: dict[str, np.ndarray], group_mask: np.ndarray,
               dimension: str, group_1: str, group_2: str = "", *, outcome_derived: bool = False) -> dict[str, Any]:
    idx = np.flatnonzero(group_mask)
    y = frame["y"][idx].astype(np.int8)
    n = len(idx)
    positives = int(y.sum()) if n else 0
    negatives = n - positives
    supported, reason = _support_reason(n, positives, negatives)
    cycles = np.unique(frame["cycles"][idx]).astype(int).tolist() if n else []
    origins = np.unique(frame["origin"][idx][frame["origin"][idx] >= 0]).astype(int).tolist() if n else []
    record: dict[str, Any] = {
        "view": view_name, "dimension": dimension, "group_1": group_1, "group_2": group_2,
        "outcome_derived_group": bool(outcome_derived), "n": n, "positives": positives,
        "negatives": negatives, "support_f1": supported, "support_reason": reason,
        "cycle_count": len(cycles), "cycle_codes": "|".join(map(str, cycles)),
        "origin_count": len(origins), "origin_codes": "|".join(map(str, origins)),
        "strict_clip_hit_rows": int(frame["strict_clip_hit"][idx].sum()) if n else 0,
        "cap_boundary_or_above_rows": int(frame["cap_hit_or_boundary"][idx].sum()) if n else 0,
        "wrong_exact_endpoint_rows": int((((frame["observed"][idx] == 0.0) & (y == 1))
                                          | ((frame["observed"][idx] == 1.0) & (y == 0))).sum()) if n else 0,
        "no_positive_weight_rows": int(frame["no_positive_weights"][idx].sum()) if n else 0,
        "zero_tfidf_vector_rows": int(frame["zero_tfidf_vector"][idx].sum()) if n else 0,
        "single_effective_neighbor_rows": int(frame["single_effective_neighbor"][idx].sum()) if n else 0,
        "selected_similarity_tie_rows": int(frame["selected_similarity_tie"][idx].sum()) if n else 0,
    }
    for arm in ("baseline", "observed", "shuffled"):
        for key, value in _metric_record(y, frame[arm][idx], supported).items():
            record[f"{arm}_{key}"] = value
    if n:
        base, obs, shuf = (frame[k][idx].astype(np.float64) for k in ("baseline", "observed", "shuffled"))
        llbase15, llobs15, llshuf15 = (_row_logloss(y, p, 1e-15) for p in (base, obs, shuf))
        llnbase, llnobs, llnshuf = (_row_logloss(y, p, np.finfo(np.float64).eps) for p in (base, obs, shuf))
        record.update({
            "observed_f1_delta_vs_sparse25": (record["observed_macro_f1"] - record["baseline_macro_f1"]) if supported else None,
            "observed_f1_delta_vs_shuffle": (record["observed_macro_f1"] - record["shuffled_macro_f1"]) if supported else None,
            "brier_improvement_vs_sparse25": float(np.mean(np.square(base - y) - np.square(obs - y))),
            "brier_improvement_vs_shuffle": float(np.mean(np.square(shuf - y) - np.square(obs - y))),
            "classification_error_improvement_vs_sparse25": float(np.mean((base >= .5) != y) - np.mean((obs >= .5) != y)),
            "classification_error_improvement_vs_shuffle": float(np.mean((shuf >= .5) != y) - np.mean((obs >= .5) != y)),
            "logloss_improvement_vs_sparse25_e15": float(np.mean(llbase15 - llobs15)),
            "logloss_improvement_vs_shuffle_e15": float(np.mean(llshuf15 - llobs15)),
            "logloss_improvement_vs_sparse25_native_epsilon": float(np.mean(llnbase - llnobs)),
            "logloss_improvement_vs_shuffle_native_epsilon": float(np.mean(llnshuf - llnobs)),
        })
    else:
        for key in ("observed_f1_delta_vs_sparse25", "observed_f1_delta_vs_shuffle",
                    "brier_improvement_vs_sparse25", "brier_improvement_vs_shuffle",
                    "classification_error_improvement_vs_sparse25", "classification_error_improvement_vs_shuffle",
                    "logloss_improvement_vs_sparse25_e15", "logloss_improvement_vs_shuffle_e15",
                    "logloss_improvement_vs_sparse25_native_epsilon", "logloss_improvement_vs_shuffle_native_epsilon"):
            record[key] = None
    return record


def build_group_records(views: dict[str, dict[str, np.ndarray]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    support_summary: dict[str, Any] = {}
    selected_views = ["full_six_primary", "new_origin_aggregate", "late", "inherited_H12_L",
                      *[f"E{o}_first4" for o in ORIGINS]]
    for origin in ORIGINS:
        views[f"E{origin}_first4"] = slice_frame(
            views[f"E{origin}_all_available"],
            (views[f"E{origin}_all_available"]["cycles"] >= ORIGIN_FIRST4[origin][0])
            & (views[f"E{origin}_all_available"]["cycles"] <= ORIGIN_FIRST4[origin][1]))
    one_way_specs = {
        "max_cosine": ("max_cosine", [0, .2, .4, .6, .8, 1], SIM_LABELS, "NO_POSITIVE_WEIGHT"),
        "ess": ("ess", [0, 4, 8, 16, 32, math.inf], ESS_LABELS, "NO_POSITIVE_WEIGHT"),
        "sparse25_probability": ("baseline_probability", [i/10 for i in range(11)], PROB_LABELS, "MISSING"),
        "mean_origin_age": ("mean_origin_age", [0, 4, 8, 16, 32, math.inf], AGE_LABELS, "NO_POSITIVE_WEIGHT"),
        "residual_sign_consistency": ("weighted_residual_sign_consistency", [0, .5, .6, .7, .8, .9, 1], SIGN_LABELS, "NO_POSITIVE_WEIGHT"),
    }
    total_rows: dict[str, Any] = {}
    for view_name in selected_views:
        frame = views[view_name]
        learned = (~frame["missing"].astype(bool)) & (frame["origin"] >= 0)
        frame_rows_before = len(records)
        dims = {}
        for name, (key, edges, labels, missing_label) in one_way_specs.items():
            cats = _fixed_bins(frame[key], edges, labels, missing_label)
            dims[name] = cats
            for label in [*labels, missing_label]:
                mask = learned & (cats == label)
                records.append(_group_row(view_name, frame, mask, name, label))
        for name in ("strict_clip_hit", "cap_hit_or_boundary", "zero_tfidf_vector", "no_positive_weights",
                     "single_effective_neighbor", "selected_similarity_tie", "saved_top32_boundary_tie"):
            values = frame[name].astype(bool)
            for value, label in ((False, "NO"), (True, "YES")):
                records.append(_group_row(view_name, frame, learned & (values == value), name, label))
        y = frame["y"].astype(np.int8)
        endpoint_wrong = (((frame["observed"] == 0.0) & (y == 1))
                          | ((frame["observed"] == 1.0) & (y == 0)))
        for value, label in ((False, "OTHER_ROWS"), (True, "WRONG_EXACT_ENDPOINT")):
            records.append(_group_row(view_name, frame, learned & (endpoint_wrong == value),
                                      "wrong_exact_endpoint_outcome_diagnostic", label,
                                      outcome_derived=True))
        sim = dims["max_cosine"]
        baseline = dims["sparse25_probability"]
        for sim_label in SIM_LABELS:
            for base_label in PROB_LABELS:
                records.append(_group_row(view_name, frame,
                                          learned & (sim == sim_label) & (baseline == base_label),
                                          "max_cosine_x_sparse25_probability", sim_label, base_label))
        broad_sim = _fixed_bins(frame["max_cosine"], [0, .4, .6, 1], SIM_BROAD_LABELS,
                                "NO_POSITIVE_WEIGHT")
        age = dims["mean_origin_age"]
        for age_label in [*AGE_LABELS, "NO_POSITIVE_WEIGHT"]:
            for sim_label in SIM_BROAD_LABELS:
                records.append(_group_row(view_name, frame,
                                          learned & (age == age_label) & (broad_sim == sim_label),
                                          "mean_origin_age_x_broad_similarity", age_label, sim_label))
        rows_view = records[frame_rows_before:]
        supported = [r for r in rows_view if r["support_f1"]]
        covered = sum(r["n"] for r in supported)
        support_summary[view_name] = {
            "learned_present_denominator": int(learned.sum()),
            "group_rows_written": len(rows_view),
            "supported_cells": len(supported),
            "unsupported_cells": len(rows_view) - len(supported),
            "supported_cell_row_counts_are_overlapping_across_dimensions": True,
            "no_outcome_based_bin_edges_used": True,
        }
        total_rows[view_name] = {"learned_present_rows": int(learned.sum()), "supported_group_cells": len(supported)}
    return records, {"view_support": support_summary, "view_denominators": total_rows}


def _anchor_checks(views: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    continuation = _json(ROOT / CONT_REPORT / "results.json")
    pilot = _json(ROOT / PILOT_REPORT / "metrics_reconstructed.json")
    pairs = {
        "W1": ("W1", "W1", "continuation"),
        "W2": ("W2", "W2", "continuation"), "W3": ("W3", "W3", "continuation"),
        "W4": ("W4", "W4", "continuation"), "W5": ("W5", "W5", "continuation"),
        "W6": ("W6", "W6", "continuation"),
        "new_origin_aggregate": ("new_origin_aggregate", "new_origin_aggregate", "continuation"),
        "full_six_primary": ("full_six_primary", "full_six_primary", "continuation"),
        "late": ("late", "late", "continuation"),
        "inherited_H12_L": ("inherited_H12_L", "inherited_H12_L", "continuation"),
        "inherited_pilot_primary_W4_W6": ("inherited_pilot_primary_W4_W6", "primary", "pilot"),
        "inherited_H12_L_pilot": ("inherited_H12_L", "H12_L", "pilot"),
    }
    checks = []
    for view_key, (frame_key, anchor_key, family) in pairs.items():
        frame = views[frame_key]
        anchor = continuation["views"][anchor_key] if family == "continuation" else pilot["metrics"][anchor_key]
        for arm in ("baseline", "observed", "shuffled"):
            saved_arm = "sparse25" if family == "pilot" and arm == "baseline" else arm
            expected = float(anchor[saved_arm]["macro_f1"])
            actual = float(_confusion(frame["y"], frame[arm])["macro_f1"])
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise RuntimeError(f"Independent Macro F1 anchor mismatch {view_key}/{arm}: {actual} vs {expected}")
            checks.append({"view": view_key, "arm": arm, "rows": len(frame["y"]),
                           "expected_macro_f1": expected, "reconstructed_macro_f1": actual,
                           "absolute_delta": abs(actual - expected), "pass": True})
    return {"status": "PASS", "checks": checks, "count": len(checks),
            "anchor_sources": [f"{CONT_REPORT}/results.json", f"{PILOT_REPORT}/metrics_reconstructed.json"]}


def _reference_math_checks(views: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    records = []
    for name in ("new_origin_aggregate", "full_six_primary", "late", "inherited_H12_L"):
        frame = views[name]
        y = frame["y"].astype(np.int8)
        out: dict[str, Any] = {"view": name, "rows": len(y)}
        for arm in ("baseline", "observed", "shuffled"):
            p = frame[arm].astype(np.float64)
            predicted = p >= THRESHOLD
            tn = int(np.sum((y == 0) & ~predicted)); fp = int(np.sum((y == 0) & predicted))
            fn = int(np.sum((y == 1) & ~predicted)); tp = int(np.sum((y == 1) & predicted))
            f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            total = tn + fp + fn + tp
            if total != len(y):
                raise RuntimeError(f"Independent confusion totals do not conserve rows for {name}/{arm}")
            out[arm] = {"confusion_sum": total, "macro_f1_reference": (f0 + f1) / 2.0,
                        "error_reference": (fp + fn) / len(y)}
            if not math.isclose(out[arm]["macro_f1_reference"], _confusion(y, p)["macro_f1"], abs_tol=1e-15, rel_tol=0):
                raise RuntimeError("Independent confusion/F1 reference disagrees with metric helper")
        baseline, observed, shuffled = (frame[k].astype(np.float64) for k in ("baseline", "observed", "shuffled"))
        brier_improvement = float(np.sum(np.square(baseline-y) - np.square(observed-y)) / len(y))
        error_improvement = float(np.sum((baseline >= .5) != y) / len(y)
                                  - np.sum((observed >= .5) != y) / len(y))
        shuffle_brier_improvement = float(np.sum(np.square(shuffled-y) - np.square(observed-y)) / len(y))
        report_add = additive_diagnostics(frame)
        if (not math.isclose(brier_improvement, report_add["brier_improvement_vs_sparse25"], abs_tol=1e-15, rel_tol=0)
                or not math.isclose(error_improvement, report_add["classification_error_improvement_vs_sparse25"], abs_tol=1e-15, rel_tol=0)
                or not math.isclose(shuffle_brier_improvement, report_add["brier_improvement_vs_shuffle"], abs_tol=1e-15, rel_tol=0)):
            raise RuntimeError(f"Independent additive loss/error math mismatch for {name}")
        out["additive_checks"] = {"brier_improvement_vs_sparse25": brier_improvement,
                                  "classification_error_improvement_vs_sparse25": error_improvement,
                                  "brier_improvement_vs_shuffle": shuffle_brier_improvement,
                                  "pass": True}
        records.append(out)
    return {"status": "PASS", "views": records}


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {"finite_n": 0, "missing_n": int(len(array)), "min": None, "p25": None,
                "median": None, "p75": None, "p90": None, "max": None}
    quantiles = np.quantile(finite, [0, .25, .5, .75, .9, 1], method="linear")
    return {"finite_n": int(len(finite)), "missing_n": int(len(array) - len(finite)),
            "min": float(quantiles[0]), "p25": float(quantiles[1]),
            "median": float(quantiles[2]), "p75": float(quantiles[3]),
            "p90": float(quantiles[4]), "max": float(quantiles[5])}


def _group_direction_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    target_views = {"full_six_primary", "new_origin_aggregate", "late", "inherited_H12_L",
                    *[f"E{o}_first4" for o in ORIGINS]}
    for view in sorted(target_views):
        subset = [r for r in records if r["view"] == view and r["support_f1"]
                  and r["dimension"] in {"max_cosine", "ess", "sparse25_probability",
                                         "mean_origin_age", "residual_sign_consistency"}]
        result[view] = {}
        for dimension in sorted({r["dimension"] for r in subset}):
            items = [r for r in subset if r["dimension"] == dimension]
            f1_deltas = [r["observed_f1_delta_vs_sparse25"] for r in items if r["observed_f1_delta_vs_sparse25"] is not None]
            brier = [r["brier_improvement_vs_sparse25"] for r in items if r["brier_improvement_vs_sparse25"] is not None]
            result[view][dimension] = {
                "supported_bins": len(items),
                "positive_f1_bins": int(sum(x > 1e-12 for x in f1_deltas)),
                "negative_f1_bins": int(sum(x < -1e-12 for x in f1_deltas)),
                "zero_or_near_zero_f1_bins": int(sum(abs(x) <= 1e-12 for x in f1_deltas)),
                "positive_brier_bins": int(sum(x > 0 for x in brier)),
                "negative_brier_bins": int(sum(x < 0 for x in brier)),
                "positive_and_negative_directions_are_descriptive_only": True,
            }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.generic):
            return clean(obj.item())
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _plot_heatmap(group_records: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subset = [r for r in group_records if r["view"] == "full_six_primary"
              and r["dimension"] == "max_cosine_x_sparse25_probability"]
    matrix = np.full((len(SIM_LABELS), len(PROB_LABELS)), np.nan)
    counts = np.zeros_like(matrix)
    for row in subset:
        i = SIM_LABELS.index(row["group_1"]); j = PROB_LABELS.index(row["group_2"])
        counts[i, j] = row["n"]
        if row["support_f1"]:
            matrix[i, j] = row["brier_improvement_vs_sparse25"]
    finite = matrix[np.isfinite(matrix)]
    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    if len(finite):
        bound = max(float(np.max(np.abs(finite))), 1e-8)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
        fig.colorbar(im, ax=ax, label="Mean Brier improvement vs Sparse25 (positive favors A02)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                label = f"n={int(counts[i,j])}"
                if np.isfinite(matrix[i, j]):
                    label += f"\n{matrix[i,j]:+.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7,
                        color="black" if not np.isfinite(matrix[i, j]) or abs(matrix[i, j]) < .65*bound else "white")
    else:
        ax.text(.5, .5, "No cells meet the frozen support rule", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks(range(len(PROB_LABELS)), PROB_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(SIM_LABELS)), SIM_LABELS)
    ax.set_xlabel("Sparse25 probability band")
    ax.set_ylabel("Maximum cosine band")
    ax.set_title("A02 paired Brier gain by similarity and baseline confidence\nFull-six primary, present-stream rows; unsupported cells show counts only")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {"path": _rel(out_path), "sha256": sha256_file(out_path),
            "supported_cells": int(np.isfinite(matrix).sum()), "cells": int(matrix.size)}


def _plot_age(group_records: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subset = [r for r in group_records if r["view"] == "full_six_primary"
              and r["dimension"] == "mean_origin_age_x_broad_similarity"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, constrained_layout=True)
    pairs = [("brier_improvement_vs_sparse25", "vs Sparse25"), ("brier_improvement_vs_shuffle", "vs matched shuffle")]
    for ax, (metric, title) in zip(axes, pairs):
        for sim_label in SIM_BROAD_LABELS:
            rows = [r for r in subset if r["group_2"] == sim_label and r["support_f1"]]
            lookup = {r["group_1"]: r for r in rows}
            xs, ys = [], []
            for i, age in enumerate(AGE_LABELS):
                if age in lookup:
                    xs.append(i); ys.append(lookup[age][metric])
            if xs:
                ax.plot(xs, ys, marker="o", label=sim_label)
        ax.axhline(0, color="black", linewidth=.8, linestyle="--")
        ax.set_title(title)
        ax.set_xticks(range(len(AGE_LABELS)), AGE_LABELS, rotation=30, ha="right")
        ax.set_xlabel("Weighted mean E − neighbor-cycle age")
        ax.set_ylabel("Mean Brier improvement (positive favors A02)")
        ax.legend(title="Max-cosine band")
    fig.suptitle("Observed assignment by neighbor age within broad similarity bands\nPrimary present-stream rows; supported cells only; descriptive, no CI")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {"path": _rel(out_path), "sha256": sha256_file(out_path),
            "supported_cells": sum(r["support_f1"] for r in subset), "cells": len(subset)}


def _ledger_state() -> dict[str, Any]:
    state = _load_budget()
    if state["fit_starts"] != 66 or state["fit_saved"] != 64 or state["fit_failed"] != 2:
        raise RuntimeError(f"Unexpected frozen fit accounting: {state}")
    state["cap_consumed_exactly"] = state["fit_starts"] == state["cap"]
    state["audit_new_fit_authorized"] = 0
    return state


def run_audit() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    freeze_state = verify_freezes_and_inputs()
    pilot_verification = _json(ROOT / PILOT_REPORT / "verification.json")
    continuation_verification = _json(ROOT / CONT_REPORT / "verification.json")
    if pilot_verification.get("status") != "PASS" or continuation_verification.get("status") != "PASS":
        raise RuntimeError("Preserved A02 verifier status is not PASS")
    budget = _ledger_state()
    with forbid_fit_calls():
        train = load_train()
        caches = load_c16_caches(train, freeze_state)
        origin_frames: dict[int, dict[str, np.ndarray]] = {}
        for origin in ORIGINS:
            origin_frames[origin] = build_origin_frame(origin, train, caches, freeze_state)
        views, view_provenance = build_views(train, origin_frames, caches)
        group_records, group_qa = build_group_records(views)
        anchor_qa = _anchor_checks(views)
        math_qa = _reference_math_checks(views)
        view_stats = {name: summarize_view(frame) for name, frame in views.items()}
        descriptor_stats = {}
        for name, frame in views.items():
            learned = (~frame["missing"].astype(bool)) & (frame["origin"] >= 0)
            descriptor_stats[name] = {key: _distribution(frame[key][learned]) for key in (
                "max_cosine", "similarity_mass", "ess", "shrink_factor", "weighted_residual_mean",
                "weighted_residual_sd", "abs_raw_correction", "mean_origin_age", "mean_query_age",
                "last4_cycle_mass", "last4_cycle_mass_fraction", "represented_cycle_count",
                "max_cycle_weight_concentration", "weighted_residual_sign_consistency")}
            descriptor_stats[name]["diagnostic_counts"] = {
                "learned_present_rows": int(learned.sum()),
                "zero_tfidf_vector_rows": int(frame["zero_tfidf_vector"][learned].sum()),
                "no_positive_weights_rows": int(frame["no_positive_weights"][learned].sum()),
                "single_effective_neighbor_rows": int(frame["single_effective_neighbor"][learned].sum()),
                "selected_similarity_tie_rows": int(frame["selected_similarity_tie"][learned].sum()),
                "saved_top32_boundary_tie_rows": int(frame["saved_top32_boundary_tie"][learned].sum()),
                "cap_boundary_or_above_rows": int(frame["cap_hit_or_boundary"][learned].sum()),
                "strict_clip_hit_rows": int(frame["strict_clip_hit"][learned].sum()),
                "wrong_exact_endpoint_rows": int(((((frame["observed"] == 0.0) & (frame["y"] == 1))
                                                    | ((frame["observed"] == 1.0) & (frame["y"] == 0)))[learned]).sum()),
            }
        directions = _group_direction_summary(group_records)
        missing_equality = {}
        for name, frame in views.items():
            mask = frame["missing"].astype(bool)
            missing_equality[name] = {"rows": int(mask.sum()),
                                      "candidate_equals_sparse25": bool(np.array_equal(frame["baseline"][mask], frame["observed"][mask])),
                                      "shuffle_equals_sparse25": bool(np.array_equal(frame["baseline"][mask], frame["shuffled"][mask]))}
            if not missing_equality[name]["candidate_equals_sparse25"] or not missing_equality[name]["shuffle_equals_sparse25"]:
                raise RuntimeError(f"Missing-stream fallback differs in view {name}")

        output_views = {name: views[name] for name in (
            "full_six_primary", "new_origin_aggregate", "late", "inherited_H12_L")}
        for name, frame in output_views.items():
            safe_arrays = {key: np.asarray(value) for key, value in frame.items() if key in FRAME_KEYS}
            np.savez_compressed(RUN_ROOT / f"row_level_{name}.npz", **safe_arrays)
        table_path = REPORT_ROOT / "group_tables.csv"
        pd.DataFrame(group_records).to_csv(table_path, index=False, encoding="utf-8")
        plot_one = _plot_heatmap(group_records, REPORT_ROOT / "reliability_similarity_confidence.png")
        plot_two = _plot_age(group_records, REPORT_ROOT / "age_within_similarity.png")
        fit_calls = FIT_CALLS
        if fit_calls != 0:
            raise RuntimeError(f"The audit attempted a prohibited fit call count={fit_calls}")

    source_info = {
        "official_train": {"path": TRAIN_REL, "sha256": freeze_state["train_sha256"], "rows": len(train)},
        "pilot_freeze_signature": freeze_state["pilot_freeze_signature"],
        "continuation_freeze_signature": freeze_state["continuation_freeze_signature"],
        "a02_pilot_verifier_status": pilot_verification.get("status"),
        "a02_continuation_verifier_status": continuation_verification.get("status"),
        "analysis_plan_sha256": freeze_state["analysis_plan_sha256"],
        "c16_cache_sha256": {str(k): sha256_file(ROOT / CACHE_PATHS[k]) for k in sorted(CACHE_PATHS)},
        "prediction_hashes": {f"E{o}_{arm}": sha256_file(_safe_pred_paths(o, arm) / "predictions.npz")
                              for o in ORIGINS for arm in ("observed", "shuffled")},
        "prediction_hashes_are_reported_without_personal_ids_or_raw_text": True,
    }
    primary = views["full_six_primary"]
    learned_present_primary = (~primary["missing"].astype(bool)) & (primary["origin"] >= 0)
    statistics = {
        "audit": "A02 reliability and temporal mechanism audit; no new fits",
        "status": "COMPLETED_ZERO_FIT",
        "prior_knowledge_disclosed": "Continuation anchors and one historical supported-subgroup F1 guard flag were known before this audit; this is not outcome-blind.",
        "source_integrity": source_info,
        "budget": budget,
        "views": view_stats,
        "view_provenance": view_provenance,
        "missing_stream_exact_equality": missing_equality,
        "descriptors": descriptor_stats,
        "supported_group_direction_counts": directions,
        "group_support": group_qa,
        "anchor_reconstruction": anchor_qa,
        "fresh_confusion_and_additive_loss_checks": math_qa,
        "primary_present_learned_endpoint_diagnostic": additive_diagnostics(slice_frame(primary, learned_present_primary)),
        "tie_note": "Saved-neighbor ties and ties at the selected top-32 boundary are counted among stored neighbors. Omitted-candidate ties were not recomputed via full query-bank scoring; frozen verifier provenance already checked similarity-descending/row-position tie ordering.",
        "comparison_note": "All group metrics compare the same query rows for Sparse25, observed A02, and matched shuffle. Groups based on residual descriptors use the observed-arm descriptor to assign shared rows to all three arms.",
        "ranking_note": "AUC/AP are reported alongside threshold-0.50 F1 to distinguish ranking changes from threshold/confidence effects.",
        "development_reuse_note": "Rows are reused development data; origin-specific predictions are new but views are not untouched holdouts. H12-L overlaps primary query rows and is reported separately.",
        "outcome_groups_note": "Wrong exact-endpoint groups are diagnostic, outcome-derived summaries only; they are not proposed as inference-time router features.",
        "test_sample_kaggle_access": False,
        "model_or_preprocessor_fit_count": 0,
        "candidate_probability_changes": 0,
        "official_fit_cap": "66/66 consumed; unchanged",
    }
    _write_json(REPORT_ROOT / "statistics.json", statistics)
    qa = {
        "status": "PASS",
        "checks": {
            **freeze_state["checks"],
            "pilot_and_continuation_historical_verifiers_pass": True,
            "all_six_c16_cache_routes_and_hashes_reconstructed": True,
            "all_five_saved_origin_observed_and_shuffle_bundles_validated": True,
            "train_row_id_cycle_joins_exact": True,
            "oof_bank_rows_past_only_and_origin_bound": True,
            "npz_labels_never_used_as_y_true": True,
            "all_saved_neighbor_corrections_reconstructed_without_predict_or_fit": True,
            "observed_and_shuffle_neighbor_geometry_identical": True,
            "missing_stream_exact_fallback": all(x["candidate_equals_sparse25"] and x["shuffle_equals_sparse25"]
                                                   for x in missing_equality.values()),
            "unique_view_denominators_and_overlap_rules": True,
            "f1_anchor_reconstruction": anchor_qa["status"] == "PASS",
            "fresh_confusion_and_additive_loss_math": math_qa["status"] == "PASS",
            "fit_call_guard_zero": FIT_CALLS == 0,
            "fit_cap_66_consumed_and_unchanged": budget["fit_starts"] == budget["cap"] == 66,
            "row_level_artifacts_under_gitignored_run_root": True,
            "tracked_reports_contain_no_raw_text_or_record_ids": True,
            "max_two_informative_pngs_written": True,
            "test_sample_kaggle_or_submission_access": False,
        },
        "row_level_outputs_ignored": [_rel(RUN_ROOT / f"row_level_{name}.npz") for name in output_views],
        "tracked_outputs": [_rel(REPORT_ROOT / name) for name in
                             ("AUDIT_PLAN.md", "statistics.json", "group_tables.csv",
                              "reliability_similarity_confidence.png", "age_within_similarity.png", "QA.json")],
        "group_tables_sha256": sha256_file(table_path),
        "plots": [plot_one, plot_two],
        "fit_calls_observed": FIT_CALLS,
        "ledger_sha256": budget["ledger_sha256"],
        "fit_starts_saved_failed": [budget["fit_starts"], budget["fit_saved"], budget["fit_failed"]],
        "known_unknowns": [
            "Kaggle screenshot/public score is not part of this audit and was not queried.",
            "The top-32 neighborhood was not recomputed from all candidates; existing frozen verifier tie-order checks were relied on.",
            "Per-row derived targets/features are stored only in the ignored run namespace, not tracked reports.",
        ],
    }
    _write_json(REPORT_ROOT / "QA.json", qa)
    return {"statistics": statistics, "qa": qa}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run_audit()
    summary = result["statistics"]
    print(json.dumps({
        "status": summary["status"],
        "budget": summary["budget"],
        "primary": summary["views"]["full_six_primary"],
        "new_origin_aggregate": summary["views"]["new_origin_aggregate"],
        "late": summary["views"]["late"],
        "H12_L": summary["views"]["inherited_H12_L"],
        "qa": result["qa"]["status"],
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
