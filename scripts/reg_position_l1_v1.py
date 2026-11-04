"""Bounded, resumable runner for REG_POSITION_L1_V1.

The official path reads only Regression Train features and strictly-past OOF
donor targets before all six heads are durable. Query Train targets are opened
only by the separate score phase after independent prelabel verification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_models import _text_values  # noqa: E402
from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block  # noqa: E402
from src.models.reg_position_l1_v1 import compose_residual_prediction, head_design, unigram_bag  # noqa: E402
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS  # noqa: E402

CARD_ID = "REG_POSITION_L1_V1"
ORIGINS = (67, 75)
ARMS = ("BAG", "REAL", "SHUFFLED")
FIT_IDS = tuple(f"{CARD_ID}_E{origin}_{arm}" for origin in ORIGINS for arm in ARMS)
RUN_REL = Path("experiments/runs/reg_position_l1_v1")
REPORT_REL = Path("reports/optimization_v3/reg_position_l1_v1")
DESIGN_REL = Path("reports/optimization_v3/regression_position_frontier_v1")
TRAIN_REL = Path("data/raw/regression/Train.csv")
EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
THREADS = 2
MAX_RSS_BYTES = 6 * 1024**3
MIN_AVAILABLE_BEFORE_FIT = 6 * 1024**3
MIN_AVAILABLE_DURING_FIT = 2 * 1024**3
ALPHA = 0.25
DELTA_CAP = 5.0
TARGET_RANGE = (0.0, 100.0)
LEDGER_CAP = 6
FEATURE_COLUMNS = ("record_id", "cycle_code", "signal_stream", "impact_load", "text_load", "entry_friction")
CODE_PATHS = (
    "scripts/reg_position_l1_v1.py",
    "scripts/verify_reg_position_l1_v1.py",
    "src/models/reg_position_l1_v1.py",
    "src/models/regression_research_tabular.py",
    "src/models/classification_models.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "tests/test_reg_position_l1_v1.py",
    "reports/optimization_v3/reg_position_l1_v1/RECIPE.json",
    "reports/optimization_v3/regression_position_frontier_v1/PROTOCOL.md",
    "reports/optimization_v3/regression_position_frontier_v1/READINESS_AND_ERROR_AUDIT.json",
    "scripts/regression_position_frontier_audit.py",
)
OOF_CACHE_REL = {
    55: Path("experiments/runs/optimization_v3_batch1/R06/R06_fit55.npz"),
    59: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit59.npz"),
    63: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit63.npz"),
    67: Path("experiments/runs/optimization_v3_batch1/R06/R06_fit67.npz"),
    71: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit71.npz"),
    75: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit75.npz"),
}


class PilotError(RuntimeError):
    pass


@dataclass
class PreparedOrigin:
    origin: int
    fit_rows: np.ndarray
    fit_ids: np.ndarray
    fit_cycles: np.ndarray
    fit_anchor: np.ndarray
    fit_residual: np.ndarray
    fit_designs: dict[str, sparse.csr_matrix]
    query_rows: np.ndarray
    query_ids: np.ndarray
    query_cycles: np.ndarray
    query_anchor: np.ndarray
    query_missing: np.ndarray
    query_designs: dict[str, sparse.csr_matrix]
    group_features: dict[str, np.ndarray]
    group_edges: dict[str, list[float]]
    provenance: dict[str, Any]


def _select_final_r06_baseline(bundle: Any) -> np.ndarray:
    """Select only the final mapped R06 member from a target-free bundle."""
    if "baseline_prediction" not in bundle.files:
        raise PilotError("R06 bundle lacks its final baseline_prediction member")
    baseline = np.asarray(bundle["baseline_prediction"], dtype=np.float64)
    if baseline.ndim != 1 or not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100)):
        raise PilotError("Final R06 baseline_prediction is invalid")
    return baseline


def _validate_donor_temporal(donor_rows: np.ndarray, donor_cycles: np.ndarray,
                             donor_cutoffs: np.ndarray, query_rows: np.ndarray,
                             origin: int, frame_rows: int) -> None:
    rows = np.asarray(donor_rows, dtype=np.int64)
    cycles = np.asarray(donor_cycles, dtype=np.int64)
    cutoffs = np.asarray(donor_cutoffs, dtype=np.int64)
    query = np.asarray(query_rows, dtype=np.int64)
    if (rows.ndim != 1 or cycles.shape != rows.shape or cutoffs.shape != rows.shape
            or len(set(rows.tolist())) != len(rows) or np.any((rows < 0) | (rows >= frame_rows))
            or np.intersect1d(rows, query).size or np.any(cycles < 56) or np.any(cycles > origin)
            or np.any(cutoffs >= cycles)):
        raise PilotError(f"E{origin} donor rows/cycles/cutoffs fail strict-past temporal validation")
    expected_cutoffs = ((cycles - 56) // 4) * 4 + 55
    if not np.array_equal(expected_cutoffs.astype(np.int64), cutoffs):
        raise PilotError(f"E{origin} donor cutoff is not the exact four-cycle OOF block")


def _validate_query_alignment(expected_origin: int, actual_origin: int,
                              actual_rows: np.ndarray, actual_ids: np.ndarray,
                              actual_cycles: np.ndarray, actual_baseline: np.ndarray,
                              expected_rows: np.ndarray, expected_ids: np.ndarray,
                              expected_cycles: np.ndarray, expected_baseline: np.ndarray) -> None:
    if (actual_origin != expected_origin
            or not np.array_equal(actual_rows, expected_rows)
            or not np.array_equal(np.asarray(actual_ids, dtype=str), np.asarray(expected_ids, dtype=str))
            or not np.array_equal(actual_cycles, expected_cycles)
            or not np.array_equal(actual_baseline, expected_baseline)):
        raise PilotError(f"E{expected_origin} query origin, rows, IDs, cycles or final R06 anchor mismatch")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    header = _canonical({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header)
    if array.dtype.kind in "OUS":
        digest.update(_canonical(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _sha_raw_array(value: Any, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _sha_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return _canonical_sha({
        "shape": list(matrix.shape),
        "data_sha256": _sha_array(matrix.data),
        "indices_sha256": _sha_array(matrix.indices),
        "indptr_sha256": _sha_array(matrix.indptr),
    })


def _same_csr(left: sparse.spmatrix, right: sparse.spmatrix, *, atol: float = 0.0) -> bool:
    a = sparse.csr_matrix(left, dtype=np.float64).copy()
    b = sparse.csr_matrix(right, dtype=np.float64).copy()
    a.sum_duplicates(); a.sort_indices(); b.sum_duplicates(); b.sort_indices()
    return (a.shape == b.shape and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices)
            and np.allclose(a.data, b.data, rtol=0.0, atol=atol))


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=0)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_event(ledger: Path, event: dict[str, Any], *, output_root: Path,
                  source_root: Path | None = None) -> None:
    ledger = ledger.resolve(); output_root = output_root.resolve()
    if output_root not in ledger.parents:
        raise PilotError("Ledger path must be strictly inside this lane output root")
    if source_root is not None and (ledger == source_root.resolve() or source_root.resolve() in ledger.parents):
        raise PilotError("Lane ledger cannot be inside the read-only source root")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(event) + b"\n"
    fd = os.open(ledger, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload); os.fsync(fd)
    finally:
        os.close(fd)


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except Exception as exc:
            raise PilotError(f"Invalid isolated ledger line {line_no}") from exc
        if not isinstance(value, dict):
            raise PilotError(f"Invalid isolated ledger event at line {line_no}")
        events.append(value)
    return events


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _resolved_parameters(*, synthetic: bool = False) -> dict[str, Any]:
    if synthetic:
        params = {**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"],
                  "n_estimators": 8, "num_leaves": 7, "min_child_samples": 4}
    else:
        params = {**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"]}
    resolved = LGBMRegressor(**params).get_params(deep=False)
    return {key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in sorted(resolved.items())}


def _contained_file(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise PilotError(f"Input path escapes its declared root: {relative}")
    if not path.is_file():
        raise PilotError(f"Required input file is missing: {relative}")
    return path


def _read_targets_for_rows(path: Path, wanted_rows: np.ndarray,
                           expected_ids: np.ndarray) -> np.ndarray:
    """Read target cells only for explicitly selected donor or post-freeze query rows."""
    rows = np.asarray(wanted_rows, dtype=np.int64)
    ids = np.asarray(expected_ids, dtype=str)
    if rows.ndim != 1 or ids.shape != rows.shape or len(set(rows.tolist())) != len(rows):
        raise PilotError("Requested target rows and IDs must be unique aligned vectors")
    wanted = {int(row): index for index, row in enumerate(rows)}
    values = np.full(len(rows), np.nan, dtype=np.float64)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        try:
            id_col = header.index("record_id")
            target_col = header.index("threat_rank")
        except ValueError as exc:
            raise PilotError("Regression Train lacks required ID/target columns") from exc
        for row_index, row in enumerate(reader):
            slot = wanted.get(row_index)
            if slot is None:
                continue
            if len(row) != len(header) or row[id_col] != ids[slot]:
                raise PilotError(f"Target row identity mismatch at source row {row_index}")
            values[slot] = float(row[target_col])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise PilotError("Selected Regression targets must be finite and within [0,100]")
    return values


def _prefix_quartiles(frame: pd.DataFrame, prefix_rows: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    result: dict[str, np.ndarray] = {}
    edges_out: dict[str, list[float]] = {}
    for name in ("impact_load", "text_load"):
        prefix = pd.to_numeric(frame.iloc[prefix_rows][name], errors="coerce").to_numpy(dtype=np.float64)
        query = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(prefix).all() or not np.isfinite(query).all():
            raise PilotError(f"{name} must be finite for frozen fit-only quartiles")
        quantiles = np.quantile(prefix, [0.25, 0.5, 0.75], method="linear")
        edges = np.unique(quantiles).astype(float)
        edges_out[name] = edges.tolist()
        result[name] = np.digitize(query, edges, right=False).astype(np.int8)
    return result, edges_out


def _load_frame(source_root: Path, readiness: dict[str, Any]) -> pd.DataFrame:
    train_path = _contained_file(source_root, TRAIN_REL)
    if _sha_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise PilotError("Official Regression Train SHA-256 differs from the frozen source")
    frame = pd.read_csv(train_path, usecols=list(FEATURE_COLUMNS), dtype={"record_id": "string"})
    frame = frame.loc[:, list(FEATURE_COLUMNS)]
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    if (len(frame) != 50_608 or not np.isfinite(pd.to_numeric(frame.cycle_code, errors="coerce")).all()
            or len(set(ids.tolist())) != len(ids)):
        raise PilotError("Official Regression feature rows, cycles or IDs failed integrity checks")
    frame["cycle_code"] = pd.to_numeric(frame.cycle_code, errors="raise").astype(np.int64)
    for name in ("impact_load", "text_load", "entry_friction"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not frame.entry_friction.dropna().isin([0, 1]).all():
        raise PilotError("entry_friction contains values outside the frozen {0,1} schema")
    return frame


def _read_oof_cache(source_root: Path, frame: pd.DataFrame, cutoff: int) -> dict[int, float]:
    path = _contained_file(source_root, OOF_CACHE_REL[cutoff])
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive["score_rows"], dtype=np.int64)
        ids = np.asarray(archive["score_record_ids"], dtype=str)
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
    if (rows.ndim != 1 or ids.shape != rows.shape or prediction.shape != rows.shape
            or len(set(rows.tolist())) != len(rows) or not np.isfinite(prediction).all()
            or np.any((prediction < 0) | (prediction > 100))):
        raise PilotError(f"R06 OOF cache for cutoff {cutoff} has invalid final predictions")
    all_ids = frame.record_id.astype(str).to_numpy(dtype=str)
    all_cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    if np.any((rows < 0) | (rows >= len(frame))) or not np.array_equal(all_ids[rows], ids):
        raise PilotError(f"R06 OOF cache row/ID identity differs for cutoff {cutoff}")
    expected_ranges = {55: (56, 67), 59: (60, 63), 63: (64, 67),
                       67: (68, 79), 71: (72, 75), 75: (76, 79)}
    lower, upper = expected_ranges[cutoff]
    if not np.all((all_cycles[rows] >= lower) & (all_cycles[rows] <= upper)):
        raise PilotError(f"R06 cache cycles are unexpected for cutoff {cutoff}")
    return {int(row): float(value) for row, value in zip(rows, prediction)}


def _manifest_path(source_root: Path, manifest: dict[str, Any], key: str) -> Path:
    item = manifest.get("matrices", {}).get(key)
    if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
        raise PilotError(f"Positional manifest lacks {key}")
    path = _contained_file(source_root, item["path"])
    if _sha_file(path) != item["sha256"]:
        raise PilotError(f"Positional cache hash mismatch: {key}")
    return path


def _verify_text_state(source_root: Path, origin: int, prefix_rows: np.ndarray,
                       frame: pd.DataFrame) -> Any:
    run = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V2_1/text_space_replays") / f"E{origin}"
    vector_path = _contained_file(source_root, run / "vectorizer.joblib")
    state_path = _contained_file(source_root, run / "text_state.json")
    receipt_path = _contained_file(source_root, run / "receipt.json")
    receipt = _json(receipt_path)
    state = _json(state_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("origin_cycle") != origin or receipt.get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY"
            or receipt.get("fit_id") != f"RNBR_IMPACT_MATCH_REPLAY_V21_E{origin}_TFIDF"
            or receipt.get("query_targets_read") is not False
            or receipt.get("classification_data_read") is not False
            or receipt.get("full_a02_model_loaded") is not False
            or receipt.get("fit_transform_calls") != 1
            or receipt.get("receipt_sha256") != _canonical_sha(body)
            or receipt.get("vectorizer_sha256") != _sha_file(vector_path)
            or receipt.get("text_state_sha256") != _sha_file(state_path)):
        raise PilotError(f"E{origin} standalone target-free vectorizer receipt failed")
    audit = receipt.get("text_audit", {})
    state_audit = state.get("audit", {})
    prefix_docs, _ = _text_values(frame.iloc[prefix_rows].signal_stream)
    docs_sha = _canonical_sha(prefix_docs)
    rows_sha = _sha_raw_array(prefix_rows, dtype="<i8")
    expected_cycle_bounds = [int(frame.cycle_code.iloc[prefix_rows].min()), origin]
    if (state.get("origin_cycle") != origin or state_audit != audit
            or receipt.get("fit_population", {}).get("rows") != len(prefix_rows)
            or receipt.get("fit_population", {}).get("cycle_bounds") != [47, origin]
            or receipt.get("fit_population", {}).get("row_positions_sha256") != rows_sha
            or receipt.get("fit_population", {}).get("documents_sha256") != docs_sha
            or audit.get("fit_rows") != len(prefix_rows)
            or audit.get("fit_row_positions_sha256") != rows_sha
            or audit.get("fit_documents_sha256") != docs_sha
            or expected_cycle_bounds[0] != 47):
        raise PilotError(f"E{origin} text prefix/document identity differs from its receipt")
    vectorizer = joblib.load(vector_path)
    vocab = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    vocab_sha = _canonical_sha([(str(term), int(index)) for term, index in vocab])
    idf_sha = _sha_raw_array(vectorizer.idf_, dtype="<f8")
    params = vectorizer.get_params(deep=False)
    expected_params = audit.get("tfidf_parameters")
    params_cmp = dict(params)
    params_cmp["dtype"] = np.dtype(params_cmp["dtype"]).name
    if (vocab_sha != audit.get("vocabulary_sha256") or idf_sha != audit.get("idf_sha256")
            or len(vectorizer.vocabulary_) != audit.get("tfidf_vocabulary_size")
            or expected_params is None or params_cmp.get("ngram_range") != tuple(expected_params.get("ngram_range", []))
            or params_cmp.get("max_features") != expected_params.get("max_features")
            or params_cmp.get("min_df") != expected_params.get("min_df")
            or params_cmp.get("max_df") != expected_params.get("max_df")
            or params_cmp.get("sublinear_tf") != expected_params.get("sublinear_tf")
            or params_cmp.get("lowercase") != expected_params.get("lowercase")
            or params_cmp.get("strip_accents") != expected_params.get("strip_accents")
            or params_cmp.get("dtype") != expected_params.get("dtype")
            or vectorizer.norm != audit.get("tfidf_norm")):
        raise PilotError(f"E{origin} standalone vectorizer vocabulary/IDF/parameters failed")
    return vectorizer


def _sample_position_reconstruction(frame: pd.DataFrame, rows: np.ndarray,
                                    cached: sparse.csr_matrix, vectorizer: Any,
                                    *, shuffled: bool, seed: int) -> dict[str, Any]:
    documents, missing = _text_values(frame.iloc[rows].signal_stream)
    absent = np.flatnonzero(missing)
    present = np.flatnonzero(~missing)
    take_absent = absent[:1]
    if len(present) < 63:
        raise PilotError("Need 63 present text rows for deterministic positional reconstruction")
    rng = np.random.default_rng(seed)
    take_present = np.sort(rng.choice(present, size=63, replace=False))
    selected = np.sort(np.concatenate([take_absent, take_present])).astype(np.int64)
    rebuilt, _ = build_position_block([documents[i] for i in selected], vectorizer,
                                      shuffled=shuffled, seed=42)
    reference = sparse.csr_matrix(cached[selected], dtype=np.float64)
    if not _same_csr(rebuilt, reference, atol=1e-12):
        raise PilotError("64-row positional tokenizer/IDF/channel reconstruction mismatch")
    return {"rows": int(len(selected)), "includes_missing": bool(len(take_absent)),
            "arm": "SHUFFLED" if shuffled else "REAL", "cached_logical_sha256": _sha_csr(reference),
            "rebuilt_logical_sha256": _sha_csr(rebuilt), "equal_atol_1e-12": True}


def _load_official_origin(source_root: Path, frame: pd.DataFrame,
                          readiness: dict[str, Any], origin: int) -> PreparedOrigin:
    run = Path("experiments/runs/optimization_v3_research_next_v1")
    folder = "recovered_bundles" if origin == 67 else "bundles"
    bundle_rel = run / "RNBR_PILOT_OFFICIAL_V1" / folder / f"RNBR_PILOT_E{origin}_OBSERVED/arrays.npz"
    bundle_path = _contained_file(source_root, bundle_rel)
    audited = readiness["origins"][str(origin)]
    if audited.get("baseline_bundle_path") != bundle_rel.as_posix() or _sha_file(bundle_path) != audited.get("baseline_bundle_sha256"):
        raise PilotError(f"E{origin} R06 anchor bundle differs from the zero-fit audit")
    with np.load(bundle_path, allow_pickle=False) as z:
        query_rows = np.asarray(z["score_rows"], dtype=np.int64)
        query_ids = np.asarray(z["score_record_ids"], dtype=str)
        query_cycles = np.asarray(z["score_cycles"], dtype=np.int64)
        # This is the frozen final R06 policy, not the bundle's candidate prediction/raw score.
        query_anchor = _select_final_r06_baseline(z)
        donor_rows = np.asarray(z["donor_rows"], dtype=np.int64)
        donor_ids = np.asarray(z["donor_record_ids"], dtype=str)
        donor_cycles = np.asarray(z["donor_cycles"], dtype=np.int64)
        donor_cutoffs = np.asarray(z["donor_source_cutoffs"], dtype=np.int64)
        donor_anchor = np.asarray(z["donor_r06_predictions"], dtype=np.float64)
        donor_residual = np.asarray(z["donor_residuals"], dtype=np.float64)
    all_ids = frame.record_id.astype(str).to_numpy(dtype=str)
    all_cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    _validate_donor_temporal(donor_rows, donor_cycles, donor_cutoffs, query_rows, origin, len(frame))
    if (query_rows.ndim != 1 or query_ids.shape != query_rows.shape or query_cycles.shape != query_rows.shape
            or query_anchor.shape != query_rows.shape or len(set(query_rows.tolist())) != len(query_rows)
            or np.any((query_rows < 0) | (query_rows >= len(frame)))
            or not np.array_equal(all_ids[query_rows], query_ids)
            or not np.array_equal(all_cycles[query_rows], query_cycles)
            or not np.all(query_cycles > origin) or not np.isfinite(query_anchor).all()
            or np.any((query_anchor < 0) | (query_anchor > 100))):
        raise PilotError(f"E{origin} query baseline rows/IDs/cycles are invalid")
    if (donor_rows.ndim != 1 or donor_ids.shape != donor_rows.shape or donor_cycles.shape != donor_rows.shape
            or donor_cutoffs.shape != donor_rows.shape or donor_anchor.shape != donor_rows.shape
            or donor_residual.shape != donor_rows.shape or len(set(donor_rows.tolist())) != len(donor_rows)
            or np.any((donor_rows < 0) | (donor_rows >= len(frame)))
            or not np.array_equal(all_ids[donor_rows], donor_ids)
            or not np.array_equal(all_cycles[donor_rows], donor_cycles)
            or np.intersect1d(donor_rows, query_rows).size
            or np.any(donor_cycles > origin) or np.any(donor_cycles < 56)
            or np.any(donor_cutoffs >= donor_cycles) or np.any(~np.isfinite(donor_anchor))
            or np.any(~np.isfinite(donor_residual))):
        raise PilotError(f"E{origin} donor rows/IDs/cycles/cutoffs failed strict-past validation")
    expected_cutoffs = ((donor_cycles - 56) // 4) * 4 + 55
    if not np.array_equal(expected_cutoffs.astype(np.int64), donor_cutoffs):
        raise PilotError(f"E{origin} donor source cutoff is not the frozen four-cycle OOF block")

    cache_map: dict[int, dict[int, float]] = {}
    for cutoff in np.unique(donor_cutoffs).tolist():
        cache_map[int(cutoff)] = _read_oof_cache(source_root, frame, int(cutoff))
    verified_donor_anchor = np.empty_like(donor_anchor)
    for idx, (row, cutoff) in enumerate(zip(donor_rows, donor_cutoffs)):
        cached = cache_map[int(cutoff)].get(int(row))
        if cached is None:
            raise PilotError(f"R06 fit-{cutoff} lacks donor row {row}")
        verified_donor_anchor[idx] = cached
    if not np.array_equal(verified_donor_anchor, donor_anchor):
        raise PilotError(f"E{origin} stored donor residual baseline is not the final R06 OOF prediction")
    donor_y = _read_targets_for_rows(_contained_file(source_root, TRAIN_REL), donor_rows, donor_ids)
    if not np.allclose(donor_y - verified_donor_anchor, donor_residual, rtol=0.0, atol=1e-12):
        raise PilotError(f"E{origin} stored donor residual differs from target minus final R06 OOF anchor")

    final_cache = _read_oof_cache(source_root, frame, origin)
    if not np.array_equal(np.asarray([final_cache.get(int(row), np.nan) for row in query_rows]), query_anchor):
        raise PilotError(f"E{origin} query anchor is not the final mapped R06 prediction")

    prefix_rows = np.flatnonzero(all_cycles <= origin).astype(np.int64)
    if len(prefix_rows) != (29_705 if origin == 67 else 42_117):
        raise PilotError(f"E{origin} prefix row count differs from the frozen fit population")
    manifest_rel = run / "POSITIONAL_TEXT_PILOT_V1" / f"MATRIX_MANIFEST_E{origin}.json"
    manifest_path = _contained_file(source_root, manifest_rel)
    if _sha_file(manifest_path) != audited.get("positional_manifest_sha256"):
        raise PilotError(f"E{origin} positional manifest differs from the zero-fit audit")
    manifest = _json(manifest_path)
    if manifest.get("origin") != origin:
        raise PilotError(f"E{origin} positional manifest has the wrong origin")
    identity_rel = run / "POSITIONAL_TEXT_PILOT_V1" / f"E{origin}/predictions_E{origin}.npz"
    identity_path = _contained_file(source_root, identity_rel)
    if _sha_file(identity_path) != audited.get("positional_identity_bundle_sha256"):
        raise PilotError(f"E{origin} positional identity archive hash differs from the zero-fit audit")
    with np.load(identity_path, allow_pickle=False) as identity:
        # Deliberately read metadata only; never index probability or label members.
        id_fit_rows = np.asarray(identity["fit_rows"], dtype=np.int64)
        id_score_rows = np.asarray(identity["score_rows"], dtype=np.int64)
        id_score_ids = np.asarray(identity["score_record_ids"], dtype=str)
        id_score_cycles = np.asarray(identity["score_cycles"], dtype=np.int64)
        id_present = np.asarray(identity["present_mask"], dtype=bool)
        id_manifest_sha = str(np.asarray(identity["matrix_manifest_sha256"]).reshape(-1)[0])
    _validate_query_alignment(origin, int(manifest.get("origin", -1)), query_rows, query_ids,
                              query_cycles, query_anchor, id_score_rows, id_score_ids,
                              id_score_cycles, query_anchor)
    texts, missing_all = _text_values(frame.signal_stream)
    if (not np.array_equal(id_fit_rows, prefix_rows) or not np.array_equal(id_score_rows, query_rows)
            or not np.array_equal(id_score_ids, query_ids) or not np.array_equal(id_score_cycles, query_cycles)
            or not np.array_equal(id_present, ~missing_all[query_rows]) or id_manifest_sha != _sha_file(manifest_path)):
        raise PilotError(f"E{origin} target-free positional row metadata differs from official Regression Train")

    base_fit_path = _manifest_path(source_root, manifest, "fit_base")
    base_score_path = _manifest_path(source_root, manifest, "score_base")
    base_fit = sparse.load_npz(base_fit_path).tocsr()
    base_score = sparse.load_npz(base_score_path).tocsr()
    if base_fit.shape[0] != len(prefix_rows) or base_score.shape[0] != len(query_rows):
        raise PilotError(f"E{origin} base matrix row count mismatch")
    width = int(base_fit.shape[1])
    if base_score.shape[1] != width:
        raise PilotError(f"E{origin} base matrix widths differ")
    vectorizer = _verify_text_state(source_root, origin, prefix_rows, frame)
    suffixes: dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    reconstructions: dict[str, dict[str, Any]] = {}
    for arm, fit_key, score_key in (
        ("REAL", "fit_position", "score_position"),
        ("SHUFFLED", "fit_shuffled_position", "score_shuffled_position"),
    ):
        fit_aug = sparse.load_npz(_manifest_path(source_root, manifest, fit_key)).tocsr()
        score_aug = sparse.load_npz(_manifest_path(source_root, manifest, score_key)).tocsr()
        if (fit_aug.shape[0] != len(prefix_rows) or score_aug.shape[0] != len(query_rows)
                or fit_aug.shape[1] <= width or score_aug.shape[1] <= width
                or fit_aug.shape[1] - width != score_aug.shape[1] - width
                or (fit_aug.shape[1] - width) % 3):
            raise PilotError(f"E{origin} augmented positional matrix shapes are invalid for {arm}")
        if (not _same_csr(fit_aug[:, :width], base_fit) or not _same_csr(score_aug[:, :width], base_score)):
            raise PilotError(f"E{origin} {arm} augmented matrix base prefix does not match base cache")
        fit_p = fit_aug[:, width:].tocsr()
        score_p = score_aug[:, width:].tocsr()
        if (not np.isfinite(fit_p.data).all() or not np.isfinite(score_p.data).all()
                or fit_p[missing_all[prefix_rows]].nnz or score_p[missing_all[query_rows]].nnz):
            raise PilotError(f"E{origin} {arm} positional matrix has invalid or missing-row features")
        fit_sample = _sample_position_reconstruction(frame, prefix_rows, fit_p, vectorizer,
                                                      shuffled=arm == "SHUFFLED", seed=4200 + origin)
        score_sample = _sample_position_reconstruction(frame, query_rows, score_p, vectorizer,
                                                        shuffled=arm == "SHUFFLED", seed=5200 + origin)
        suffixes[arm] = (fit_p, score_p)
        reconstructions[arm] = {"fit": fit_sample, "query": score_sample}
        manifest_hashes = {
            "fit_base": manifest["matrices"]["fit_base"]["sha256"],
            "score_base": manifest["matrices"]["score_base"]["sha256"],
            fit_key: manifest["matrices"][fit_key]["sha256"],
            score_key: manifest["matrices"][score_key]["sha256"],
        }
        audited_matrices = audited["matrices"]
        for key in (fit_key, score_key):
            if audited_matrices[key]["sha256"] != manifest_hashes[key]:
                raise PilotError(f"E{origin} {key} hash differs from the readiness audit")
    fit_docs = [texts[int(row)] for row in donor_rows]
    query_docs = [texts[int(row)] for row in query_rows]
    bag_fit = unigram_bag(vectorizer, fit_docs)
    bag_query = unigram_bag(vectorizer, query_docs)
    prefix_positions = {int(row): idx for idx, row in enumerate(prefix_rows)}
    donor_positions = np.asarray([prefix_positions[int(row)] for row in donor_rows], dtype=np.int64)
    fit_designs = {"BAG": head_design(bag_fit, None, donor_anchor)}
    query_designs = {"BAG": head_design(bag_query, None, query_anchor)}
    for arm in ("REAL", "SHUFFLED"):
        fit_p, score_p = suffixes[arm]
        fit_designs[arm] = head_design(bag_fit, fit_p[donor_positions], donor_anchor)
        query_designs[arm] = head_design(bag_query, score_p, query_anchor)
    group_features, edges = _prefix_quartiles(frame, prefix_rows)
    texts_meta = {
        "fit_source_rows": int(len(donor_rows)), "query_source_rows": int(len(query_rows)),
        "unigram_features": int(bag_fit.shape[1]),
        "position_features": {arm: int(suffixes[arm][0].shape[1]) for arm in ("REAL", "SHUFFLED")},
        "prefix_quartile_edges": edges,
        "position_reconstruction": reconstructions,
    }
    provenance = {
        "origin": origin, "fit_rows": int(len(donor_rows)), "query_rows": int(len(query_rows)),
        "fit_row_ids_sha256": _sha_array(donor_ids), "query_row_ids_sha256": _sha_array(query_ids),
        "fit_rows_source_positions_sha256": _sha_array(donor_rows, dtype="<i8"),
        "query_rows_source_positions_sha256": _sha_array(query_rows, dtype="<i8"),
        "fit_cutoff_range": [int(donor_cycles.min()), int(donor_cycles.max())],
        "fit_source_cutoff_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(donor_cutoffs, return_counts=True))},
        "donor_final_r06_prediction_sha256": _sha_array(verified_donor_anchor, dtype="<f8"),
        "donor_residual_sha256": _sha_array(donor_residual, dtype="<f8"),
        "query_final_r06_prediction_sha256": _sha_array(query_anchor, dtype="<f8"),
        "baseline_member": "baseline_prediction; final mapped R06",
        "candidate_member_excluded": "prediction and raw_candidate",
        "input_reconstruction": texts_meta,
        "matrix_manifest_sha256": _sha_file(manifest_path),
    }
    return PreparedOrigin(
        origin=origin, fit_rows=donor_rows, fit_ids=donor_ids, fit_cycles=donor_cycles,
        fit_anchor=verified_donor_anchor, fit_residual=donor_residual,
        fit_designs=fit_designs, query_rows=query_rows, query_ids=query_ids,
        query_cycles=query_cycles, query_anchor=query_anchor, query_missing=missing_all[query_rows],
        query_designs=query_designs,
        group_features={**{k: v[query_rows] for k, v in group_features.items()},
                        "entry_friction": frame.entry_friction.to_numpy(dtype=np.float64)[query_rows]},
        group_edges=edges, provenance=provenance,
    )


def _load_official(source_root: Path) -> tuple[pd.DataFrame, dict[int, PreparedOrigin], dict[str, Any]]:
    readiness_path = ROOT / DESIGN_REL / "READINESS_AND_ERROR_AUDIT.json"
    readiness = _json(readiness_path)
    if (readiness.get("status") != "ZERO_FIT_READINESS_PASS"
            or readiness.get("classification_targets_read") is not False
            or readiness.get("test_accessed") is not False
            or readiness.get("new_fits") != 0
            or readiness.get("shared_ledger_written") is not False):
        raise PilotError("Zero-fit readiness audit is not a valid frozen input")
    frame = _load_frame(source_root, readiness)
    origins: dict[int, PreparedOrigin] = {}
    for origin in ORIGINS:
        origins[origin] = _load_official_origin(source_root, frame, readiness, origin)
        gc.collect()
    return frame, origins, readiness


def _official_input_paths(source_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    source_fixed = [TRAIN_REL, Path("reports/optimization_v3/rnbr_postpublic_v1/audit.json"),
                    Path("src/models/optimization_v3_batch1_r06.py"),
                    Path("experiments/optimization_v3/batch1/R06/recipe_revision_02.json"),
                    *OOF_CACHE_REL.values()]
    for rel in source_fixed:
        paths[f"main/{rel.as_posix()}"] = _contained_file(source_root, rel)
    run = Path("experiments/runs/optimization_v3_research_next_v1")
    for origin in ORIGINS:
        folder = "recovered_bundles" if origin == 67 else "bundles"
        paths[f"main/{(run / 'RNBR_PILOT_OFFICIAL_V1' / folder / f'RNBR_PILOT_E{origin}_OBSERVED/arrays.npz').as_posix()}"] = _contained_file(
            source_root, run / "RNBR_PILOT_OFFICIAL_V1" / folder / f"RNBR_PILOT_E{origin}_OBSERVED/arrays.npz")
        manifest_rel = run / "POSITIONAL_TEXT_PILOT_V1" / f"MATRIX_MANIFEST_E{origin}.json"
        paths[f"main/{manifest_rel.as_posix()}"] = _contained_file(source_root, manifest_rel)
        identity_rel = run / "POSITIONAL_TEXT_PILOT_V1" / f"E{origin}/predictions_E{origin}.npz"
        paths[f"main/{identity_rel.as_posix()}"] = _contained_file(source_root, identity_rel)
        manifest = _json(paths[f"main/{manifest_rel.as_posix()}"])
        for key in ("fit_base", "score_base", "fit_position", "score_position",
                    "fit_shuffled_position", "score_shuffled_position"):
            rel = Path(manifest["matrices"][key]["path"])
            paths[f"main/{rel.as_posix()}"] = _contained_file(source_root, rel)
        text_rel = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V2_1/text_space_replays") / f"E{origin}"
        for name in ("vectorizer.joblib", "text_state.json", "receipt.json"):
            rel = text_rel / name
            paths[f"main/{rel.as_posix()}"] = _contained_file(source_root, rel)
    for rel in CODE_PATHS:
        path = (ROOT / rel).resolve()
        if not path.is_file():
            raise PilotError(f"Frozen source file missing: {rel}")
        paths[f"worktree/{Path(rel).as_posix()}"] = path
    return dict(sorted(paths.items()))


def _hash_path_map(paths: dict[str, Path]) -> dict[str, str]:
    return {key: _sha_file(path) for key, path in sorted(paths.items())}


def _package_versions() -> dict[str, str]:
    import lightgbm
    import matplotlib
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__}


def _freeze_payload(source_root: Path, origins: dict[int, PreparedOrigin],
                    readiness: dict[str, Any]) -> dict[str, Any]:
    paths = _official_input_paths(source_root)
    dirty = _git("status", "--porcelain")
    if dirty:
        raise PilotError("Freeze requires a committed, clean implementation worktree")
    parameters = _resolved_parameters()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "status": "FROZEN_BEFORE_OFFICIAL_FITS",
        "source_commit_before_freeze": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "source_hashes": {key: value for key, value in _hash_path_map(paths).items() if key.startswith("worktree/")},
        "input_hashes": {key: value for key, value in _hash_path_map(paths).items() if key.startswith("main/")},
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "readiness_audit_sha256": _sha_file(ROOT / DESIGN_REL / "READINESS_AND_ERROR_AUDIT.json"),
        "origins": {str(origin): origins[origin].provenance for origin in ORIGINS},
        "parameters": parameters,
        "parameter_source": {
            "path": "src/models/regression_research_tabular.py",
            "base": "COMMON_PARAMETERS",
            "set": "PARAMETER_SETS['regularized']",
            "resolved_sha256": _canonical_sha(parameters),
        },
        "versions": _package_versions(),
        "policy": {"alpha": ALPHA, "delta_cap": DELTA_CAP, "target_range": list(TARGET_RANGE),
                   "unigram_normalize_once": True, "position_suffix_normalize": False,
                   "rank_map_after_residual": False, "query_target_open_before_all_saved": False},
        "fit_ids_order": list(FIT_IDS),
        "accounting": {"isolated_ledger": "experiments/runs/reg_position_l1_v1/fit_ledger.jsonl",
                       "pilot_cap": LEDGER_CAP, "pilot_fit_starts": LEDGER_CAP,
                       "retries": 0, "shared_main_ledger_written": False},
        "boundaries": {"classification_targets_or_probabilities_read": False,
                       "test_or_sample_accessed": False, "kaggle_accessed": False,
                       "query_targets_used_for_fit": False, "r06_refit": False,
                       "pilot_preprocessor_fits": 0},
        "readiness_summary": {"status": readiness["status"], "new_fits": readiness["new_fits"],
                              "shared_ledger_written": readiness["shared_ledger_written"]},
    }
    payload["freeze_signature"] = _canonical_sha(payload)
    return payload


def freeze_official(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if source_root == ROOT.resolve() or ROOT.resolve() in source_root.parents:
        raise PilotError("Main source root and Luna 2 worktree must remain separate")
    _, origins, readiness = _load_official(source_root)
    payload = _freeze_payload(source_root, origins, readiness)
    path = ROOT / REPORT_REL / "FREEZE.json"
    if path.exists():
        existing = _json(path)
        if existing == payload:
            return existing
        raise PilotError("A prior official freeze exists; do not overwrite it")
    _atomic_json(path, payload)
    return payload


def _load_freeze(source_root: Path) -> dict[str, Any]:
    path = ROOT / REPORT_REL / "FREEZE.json"
    freeze = _json(path)
    signature = freeze.get("freeze_signature")
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if (freeze.get("card_id") != CARD_ID or freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_FITS"
            or signature != _canonical_sha(body) or freeze.get("fit_ids_order") != list(FIT_IDS)
            or freeze.get("accounting", {}).get("pilot_cap") != LEDGER_CAP):
        raise PilotError("Official freeze schema/signature/accounting is invalid")
    paths = _official_input_paths(source_root)
    now = _hash_path_map(paths)
    expected = {**freeze.get("source_hashes", {}), **freeze.get("input_hashes", {})}
    if now != expected:
        mismatches = [key for key in sorted(set(now) | set(expected)) if now.get(key) != expected.get(key)]
        raise PilotError(f"Frozen source/input hashes changed: {mismatches[:8]}")
    if freeze.get("parameters") != _resolved_parameters():
        raise PilotError("Resolved LightGBM parameters differ from the frozen source")
    return freeze


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resource_snapshot() -> dict[str, Any]:
    process = psutil.Process()
    return {"available_ram_bytes": int(psutil.virtual_memory().available),
            "process_rss_bytes": int(process.memory_info().rss),
            "thread_limit": THREADS, "max_rss_bytes": MAX_RSS_BYTES}


def _process_identity(process: psutil.Process) -> tuple[str, bool, float]:
    try:
        cmd = process.cmdline()
        name = process.name()
        script = next((Path(token.strip('"')).name for token in cmd
                       if token.lower().endswith(".py")), None)
        if script is None:
            for index, token in enumerate(cmd[:-1]):
                if token == "-m":
                    script = f"module:{cmd[index + 1]}"
                    break
        identity = script or ("interactive-python" if len(cmd) <= 1 else "other-python")
        known_fit = any(token in " ".join(cmd).lower() for token in (
            "optimization_v3_positional_contrast_v1", "reg_position_l1_v1", "lightgbm"))
        cpu = float(process.cpu_times().user + process.cpu_times().system)
        return identity, known_fit, cpu
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return "unavailable-process-metadata", False, 0.0


def _active_training_processes() -> list[dict[str, Any]]:
    current = os.getpid()
    candidates: list[tuple[psutil.Process, str, bool, float]] = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        if proc.info.get("pid") == current or "python" not in str(proc.info.get("name", "")).lower():
            continue
        identity, known_fit, cpu = _process_identity(proc)
        candidates.append((proc, identity, known_fit, cpu))
    time.sleep(1.0)
    active: list[dict[str, Any]] = []
    for proc, identity, known_fit, before_cpu in candidates:
        try:
            after = proc.cpu_times()
            delta = float(after.user + after.system) - before_cpu
            if known_fit or delta >= 0.5:
                active.append({"pid": int(proc.pid), "identity": identity,
                               "known_fit_command": bool(known_fit), "cpu_seconds_1s": round(delta, 3)})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return active


def _wait_for_slot(*, max_wait_seconds: int = 1800, poll_seconds: int = 30) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait_seconds
    while True:
        resource = _resource_snapshot()
        trainers = _active_training_processes()
        ready = (resource["available_ram_bytes"] >= MIN_AVAILABLE_BEFORE_FIT
                 and resource["process_rss_bytes"] < MAX_RSS_BYTES and not trainers)
        snapshot = {**resource, "active_training_processes": trainers,
                    "minimum_available_before_fit_bytes": MIN_AVAILABLE_BEFORE_FIT,
                    "minimum_available_during_fit_bytes": MIN_AVAILABLE_DURING_FIT}
        if ready:
            return snapshot
        print(json.dumps({"status": "READY_WAITING_COMPUTE", "resource": snapshot}, ensure_ascii=False), flush=True)
        if time.monotonic() >= deadline:
            raise PilotError("Compute slot did not become ready within the bounded 30 minute wait")
        time.sleep(min(poll_seconds, max(1, deadline - time.monotonic())))


class _TrainingLock:
    """Windows advisory byte lock; the file is retained and never deleted."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if self.stream.read(1) == b"":
            self.stream.seek(0); self.stream.write(b"0"); self.stream.flush()
        self.stream.seek(0)
        try:
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.stream.close(); self.stream = None
            raise PilotError("This lane's training slot is locked by another active runner") from exc
        self.stream.seek(0); self.stream.truncate(); self.stream.write(str(os.getpid()).encode("ascii")); self.stream.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            import msvcrt
            self.stream.seek(0)
            try:
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.stream.close()


def _fit_artifact_paths(output_root: Path, fit_id: str) -> dict[str, Path]:
    run = output_root / RUN_REL
    return {"model": run / f"{fit_id}.joblib", "predictions": run / f"{fit_id}.npz",
            "receipt": run / f"{fit_id}.receipt.json"}


def _ledger_state(events: list[dict[str, Any]], freeze: dict[str, Any]) -> dict[str, Any]:
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    if (len(reservations) != 1 or reservations[0].get("fit_ids") != list(FIT_IDS)
            or reservations[0].get("cap") != LEDGER_CAP
            or reservations[0].get("freeze_signature") != freeze["freeze_signature"]
            or len(starts) > LEDGER_CAP or len(saved) > len(starts) or len(failed) > len(starts)):
        raise PilotError("Isolated fit ledger reservation or counts are invalid")
    if any(e.get("fit_id") not in FIT_IDS for e in starts + saved + failed):
        raise PilotError("Isolated ledger contains a fit ID outside this frozen pilot")
    if len({e.get("fit_id") for e in starts}) != len(starts):
        raise PilotError("Isolated ledger contains duplicate fit starts; no retry is allowed")
    if len({e.get("fit_id") for e in saved}) != len(saved) or len({e.get("fit_id") for e in failed}) != len(failed):
        raise PilotError("Isolated ledger has duplicate terminal events")
    return {"reservations": reservations, "starts": starts, "saved": saved, "failed": failed,
            "pending": [e for e in starts if e.get("fit_id") not in {s.get("fit_id") for s in saved + failed}]}


def _reserve_ledger(output_root: Path, source_root: Path, freeze: dict[str, Any]) -> Path:
    run = output_root / RUN_REL
    run.mkdir(parents=True, exist_ok=True)
    ledger = run / "fit_ledger.jsonl"
    events = _read_events(ledger)
    if events:
        _ledger_state(events, freeze)
        return ledger
    _append_event(ledger, {"at_utc": _utc(), "event": "CARD_FITS_RESERVED", "card_id": CARD_ID,
                           "reservation_key": f"{CARD_ID}:PILOT_6", "cap": LEDGER_CAP,
                           "fit_ids": list(FIT_IDS), "freeze_signature": freeze["freeze_signature"],
                           "execution_commit": _git("rev-parse", "HEAD"),
                           "shared_main_ledger_written": False},
                  output_root=output_root, source_root=source_root)
    return ledger


def _receipt_valid(path: Path, fit_id: str, freeze: dict[str, Any]) -> dict[str, Any]:
    receipt = _json(path)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or receipt.get("receipt_sha256") != _canonical_sha(body)):
        raise PilotError(f"Fit receipt is invalid for {fit_id}")
    return receipt


def _fit_one(origin: PreparedOrigin, arm: str, *, fit_id: str, freeze: dict[str, Any],
             output_root: Path, source_root: Path, ledger: Path, parameters: dict[str, Any],
             interrupt_after_model: bool = False, synthetic: bool = False,
             fit_invocations: list[str] | None = None,
             wait_for_slot: bool = True) -> dict[str, Any]:
    paths = _fit_artifact_paths(output_root, fit_id)
    if paths["receipt"].exists():
        receipt = _receipt_valid(paths["receipt"], fit_id, freeze)
        if _sha_file(paths["model"]) != receipt.get("model_sha256") or _sha_file(paths["predictions"]) != receipt.get("predictions_sha256"):
            raise PilotError(f"Saved artifact hash mismatch for {fit_id}")
        return receipt
    events = _read_events(ledger)
    state = _ledger_state(events, freeze)
    starts = [e for e in state["starts"] if e.get("fit_id") == fit_id]
    if any(e.get("fit_id") == fit_id for e in state["failed"]):
        raise PilotError(f"Fit {fit_id} already failed and is consumed; retry prohibited")
    params_sha = _canonical_sha(parameters)
    design_fit = origin.fit_designs[arm]
    design_query = origin.query_designs[arm]
    design_sha = _sha_csr(design_fit)
    target_sha = _sha_array(origin.fit_residual, dtype="<f8")
    query_design_sha = _sha_csr(design_query)
    if starts:
        if not paths["model"].is_file():
            raise PilotError(f"Fit {fit_id} was started without a durable model; consumed, no retry")
        payload = joblib.load(paths["model"])
        if (not isinstance(payload, dict) or payload.get("fit_id") != fit_id
                or payload.get("freeze_signature") != freeze["freeze_signature"]
                or payload.get("train_design_sha256") != design_sha
                or payload.get("target_sha256") != target_sha
                or payload.get("parameters_sha256") != params_sha):
            raise PilotError(f"Started fit {fit_id} has no valid resumable model payload")
        model = payload["model"]
        if not paths["predictions"].exists():
            head = np.asarray(model.predict(design_query), dtype=np.float64)
            prediction, delta = compose_residual_prediction(origin.query_anchor, head, origin.query_missing)
            _atomic_npz(paths["predictions"], fit_id=np.asarray([fit_id]),
                        origin=np.asarray([origin.origin], dtype=np.int64), arm=np.asarray([arm]),
                        rows=origin.query_rows, record_ids=origin.query_ids,
                        cycles=origin.query_cycles, baseline=origin.query_anchor,
                        missing=origin.query_missing, head_prediction=head,
                        delta=delta, prediction=prediction)
        model_sha = _sha_file(paths["model"]); prediction_sha = _sha_file(paths["predictions"])
        receipt = {"schema_version": 1, "fit_id": fit_id, "origin": origin.origin, "arm": arm,
                   "freeze_signature": freeze["freeze_signature"], "execution_commit": _git("rev-parse", "HEAD"),
                   "parameters": parameters, "parameters_sha256": params_sha,
                   "train_design_sha256": design_sha, "query_design_sha256": query_design_sha,
                   "target_sha256": target_sha, "fit_rows": len(origin.fit_rows),
                   "fit_row_ids_sha256": _sha_array(origin.fit_ids),
                   "model_sha256": model_sha, "predictions_sha256": prediction_sha,
                   "resumed_from_durable_model_without_refit": True,
                   "query_targets_read_for_fit": False, "test_or_sample_accessed": False,
                   "classification_labels_or_probabilities_accessed": False,
                   "shared_main_ledger_written": False}
        receipt["receipt_sha256"] = _canonical_sha(receipt)
        _atomic_json(paths["receipt"], receipt)
        _append_event(ledger, {"at_utc": _utc(), "event": "FIT_SAVED", "fit_id": fit_id,
                               "freeze_signature": freeze["freeze_signature"],
                               "model_sha256": model_sha, "predictions_sha256": prediction_sha,
                               "receipt_sha256": receipt["receipt_sha256"],
                               "shared_main_ledger_written": False,
                               "resumed_without_refit": True}, output_root=output_root, source_root=source_root)
        return receipt
    if paths["model"].exists() or paths["predictions"].exists() or paths["receipt"].exists():
        raise PilotError(f"Artifact exists without a ledger start for {fit_id}; manual audit required")

    resource = (_resource_snapshot() if synthetic else
                _wait_for_slot(max_wait_seconds=1800 if wait_for_slot else 0))
    _append_event(ledger, {"at_utc": _utc(), "event": "FIT_STARTED", "fit_id": fit_id,
                           "card_id": CARD_ID, "origin": origin.origin, "arm": arm,
                           "freeze_signature": freeze["freeze_signature"],
                           "parameters_sha256": params_sha, "train_design_sha256": design_sha,
                           "query_design_sha256": query_design_sha, "target_sha256": target_sha,
                           "fit_rows": len(origin.fit_rows), "resource_preflight": resource,
                           "shared_main_ledger_written": False,
                           "execution_commit": _git("rev-parse", "HEAD")},
                  output_root=output_root, source_root=source_root)
    started = time.perf_counter()
    model = LGBMRegressor(**parameters)
    try:
        if fit_invocations is not None:
            fit_invocations.append(fit_id)
        with threadpool_limits(limits=THREADS):
            model.fit(design_fit, origin.fit_residual)
        model_payload = {"model": model, "fit_id": fit_id, "origin": origin.origin, "arm": arm,
                         "freeze_signature": freeze["freeze_signature"], "parameters_sha256": params_sha,
                         "train_design_sha256": design_sha, "query_design_sha256": query_design_sha,
                         "target_sha256": target_sha, "fit_row_ids_sha256": _sha_array(origin.fit_ids)}
        _atomic_joblib(paths["model"], model_payload)
        if interrupt_after_model:
            raise PilotError("Synthetic interruption after durable model save, before predictions/receipt")
        train_head = np.asarray(model.predict(design_fit), dtype=np.float64)
        query_head = np.asarray(model.predict(design_query), dtype=np.float64)
        if (not np.isfinite(train_head).all() or not np.isfinite(query_head).all()
                or query_head.shape != origin.query_anchor.shape):
            raise PilotError(f"Nonfinite/misaligned fitted head output for {fit_id}")
        prediction, delta = compose_residual_prediction(origin.query_anchor, query_head, origin.query_missing)
        _atomic_npz(paths["predictions"], fit_id=np.asarray([fit_id]),
                    origin=np.asarray([origin.origin], dtype=np.int64), arm=np.asarray([arm]),
                    rows=origin.query_rows, record_ids=origin.query_ids,
                    cycles=origin.query_cycles, baseline=origin.query_anchor,
                    missing=origin.query_missing, head_prediction=query_head,
                    delta=delta, prediction=prediction)
        resource_after = _resource_snapshot()
        if (not synthetic and (resource_after["process_rss_bytes"] > MAX_RSS_BYTES
                or resource_after["available_ram_bytes"] < MIN_AVAILABLE_DURING_FIT)):
            raise PilotError(f"Post-fit resource limit exceeded for {fit_id}")
        model_sha = _sha_file(paths["model"]); prediction_sha = _sha_file(paths["predictions"])
        fit_mae = float(np.mean(np.abs(origin.fit_residual - train_head)))
        receipt = {"schema_version": 1, "fit_id": fit_id, "origin": origin.origin, "arm": arm,
                   "freeze_signature": freeze["freeze_signature"], "execution_commit": _git("rev-parse", "HEAD"),
                   "parameters": parameters, "parameters_sha256": params_sha,
                   "train_design_sha256": design_sha, "query_design_sha256": query_design_sha,
                   "target_sha256": target_sha, "fit_rows": len(origin.fit_rows),
                   "fit_row_ids_sha256": _sha_array(origin.fit_ids),
                   "in_sample_training_mae_descriptive_only": fit_mae,
                   "fit_elapsed_seconds": round(time.perf_counter() - started, 6),
                   "resource_preflight": resource, "resource_after_fit": resource_after,
                   "model_sha256": model_sha, "predictions_sha256": prediction_sha,
                   "resumed_from_durable_model_without_refit": False,
                   "synthetic_only": bool(synthetic),
                   "query_targets_read_for_fit": False, "test_or_sample_accessed": False,
                   "classification_labels_or_probabilities_accessed": False,
                   "shared_main_ledger_written": False}
        receipt["receipt_sha256"] = _canonical_sha(receipt)
        _atomic_json(paths["receipt"], receipt)
        _append_event(ledger, {"at_utc": _utc(), "event": "FIT_SAVED", "fit_id": fit_id,
                               "freeze_signature": freeze["freeze_signature"],
                               "model_sha256": model_sha, "predictions_sha256": prediction_sha,
                               "receipt_sha256": receipt["receipt_sha256"],
                               "shared_main_ledger_written": False,
                               "elapsed_seconds": receipt["fit_elapsed_seconds"]},
                      output_root=output_root, source_root=source_root)
        return receipt
    except Exception as exc:
        # A durable model is recoverable without refitting. Otherwise the started
        # fit is consumed and receives one terminal failure event.
        if paths["model"].exists():
            try:
                payload = joblib.load(paths["model"])
                recoverable = (isinstance(payload, dict) and payload.get("fit_id") == fit_id
                               and payload.get("freeze_signature") == freeze["freeze_signature"]
                               and payload.get("train_design_sha256") == design_sha
                               and payload.get("target_sha256") == target_sha)
            except Exception:
                recoverable = False
        else:
            recoverable = False
        if not recoverable:
            _append_event(ledger, {"at_utc": _utc(), "event": "FIT_FAILED", "fit_id": fit_id,
                                   "freeze_signature": freeze["freeze_signature"],
                                   "error_type": type(exc).__name__, "error": str(exc)[:500]},
                          output_root=output_root, source_root=source_root)
        raise


def run_official_fit(source_root: Path, *, wait_for_slot: bool = True) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = ROOT.resolve()
    if output_root in source_root.parents or source_root in output_root.parents or output_root == source_root:
        raise PilotError("Production output and source roots must be disjoint")
    freeze = _load_freeze(source_root)
    ledger = _reserve_ledger(output_root, source_root, freeze)
    lock = output_root / RUN_REL / "training.lock"
    if not wait_for_slot:
        # A zero-wait check is useful for a clean scheduling handoff.
        resource = _resource_snapshot()
        trainers = _active_training_processes()
        if resource["available_ram_bytes"] < MIN_AVAILABLE_BEFORE_FIT or trainers:
            raise PilotError(f"Compute slot unavailable: RAM={resource['available_ram_bytes']}; active={trainers}")
    results = []
    with _TrainingLock(lock):
        frame, origins, _ = _load_official(source_root)
        params = freeze["parameters"]
        for origin in ORIGINS:
            for arm in ARMS:
                fit_id = f"{CARD_ID}_E{origin}_{arm}"
                results.append(_fit_one(origins[origin], arm, fit_id=fit_id, freeze=freeze,
                                        output_root=output_root, source_root=source_root,
                                        ledger=ledger, parameters=params,
                                        wait_for_slot=wait_for_slot))
                gc.collect()
        del frame, origins
    events = _read_events(ledger)
    state = _ledger_state(events, freeze)
    if len(state["starts"]) != LEDGER_CAP or len(state["saved"]) != LEDGER_CAP or state["failed"] or state["pending"]:
        raise PilotError("Six-fit pilot is not fully saved in its isolated ledger")
    result = {"status": "SIX_FITS_DURABLE", "fit_ids": list(FIT_IDS),
              "freeze_signature": freeze["freeze_signature"],
              "execution_commit": _git("rev-parse", "HEAD"),
              "ledger_sha256": _sha_file(ledger), "fit_starts": len(state["starts"]),
              "fit_saved": len(state["saved"]), "fit_failed": len(state["failed"]),
              "fit_pending": len(state["pending"]),
              "query_targets_read": False, "test_or_sample_accessed": False,
              "classification_labels_or_probabilities_accessed": False}
    _atomic_json(ROOT / REPORT_REL / "PREFIT_COMPLETE.json", result)
    return result


def _load_saved_predictions(output_root: Path, fit_id: str, freeze: dict[str, Any]) -> dict[str, np.ndarray]:
    paths = _fit_artifact_paths(output_root, fit_id)
    receipt = _receipt_valid(paths["receipt"], fit_id, freeze)
    if (_sha_file(paths["model"]) != receipt.get("model_sha256")
            or _sha_file(paths["predictions"]) != receipt.get("predictions_sha256")):
        raise PilotError(f"Saved prediction/model hash mismatch for {fit_id}")
    with np.load(paths["predictions"], allow_pickle=False) as z:
        result = {key: z[key] for key in z.files}
    if str(result["fit_id"].reshape(-1)[0]) != fit_id:
        raise PilotError(f"Saved prediction has the wrong fit ID: {fit_id}")
    return result


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) == 0 or y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise PilotError("Metric inputs must be nonempty, finite and aligned")
    return float(np.mean(np.abs(y - prediction)))


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    from scipy.stats import spearmanr
    value = spearmanr(left, right).statistic
    return None if not np.isfinite(value) else float(value)


def _group_masks(origin: PreparedOrigin, y: np.ndarray) -> dict[str, np.ndarray]:
    masks = {"missing": origin.query_missing.copy(), "target_le5": y <= 5.0,
             "target_ge95": y >= 95.0,
             "EF0": np.isfinite(origin.group_features["entry_friction"])
                   & (origin.group_features["entry_friction"] == 0),
             "EF1": np.isfinite(origin.group_features["entry_friction"])
                   & (origin.group_features["entry_friction"] == 1)}
    for name in ("impact_load", "text_load"):
        bins = origin.group_features[name]
        for level in range(int(np.max(bins)) + 1):
            masks[f"{name}_Q{level + 1}"] = bins == level
    return masks


def _metric_view(origin: PreparedOrigin, y: np.ndarray, arrays: dict[str, dict[str, np.ndarray]],
                 mask: np.ndarray, *, include_groups: bool) -> dict[str, Any]:
    if mask.shape != y.shape or not mask.any():
        raise PilotError("Frozen metric view has no rows")
    baseline = arrays["BAG"]["baseline"].astype(np.float64)
    base_mae = _mae(y[mask], baseline[mask])
    by_arm: dict[str, Any] = {"R06": {"mae": base_mae, "spearman": _spearman(y[mask], baseline[mask])}}
    for arm in ARMS:
        pred = arrays[arm]["prediction"].astype(np.float64)
        head = arrays[arm]["head_prediction"].astype(np.float64)
        residual = y - baseline
        by_arm[arm] = {"mae": _mae(y[mask], pred[mask]),
                       "gain_vs_R06": base_mae - _mae(y[mask], pred[mask]),
                       "spearman_y_vs_prediction": _spearman(y[mask], pred[mask]),
                       "spearman_query_residual_vs_head": _spearman(residual[mask], head[mask]),
                       "missing_route_exact": bool(np.array_equal(pred[origin.query_missing], baseline[origin.query_missing]))}
        if arm != "R06":
            by_arm[arm]["mae_harm_vs_R06"] = by_arm[arm]["mae"] - base_mae
    for arm in ("REAL",):
        by_arm[arm]["gain_vs_BAG"] = by_arm["BAG"]["mae"] - by_arm[arm]["mae"]
        by_arm[arm]["gain_vs_SHUFFLED"] = by_arm["SHUFFLED"]["mae"] - by_arm[arm]["mae"]
    result: dict[str, Any] = {"n": int(mask.sum()), "metrics": by_arm}
    if include_groups:
        group_result: dict[str, Any] = {}
        for name, group_mask in _group_masks(origin, y).items():
            selected = mask & group_mask
            n = int(selected.sum())
            if n < 100:
                group_result[name] = {"n": n, "status": "UNKNOWN_UNSUPPORTED"}
                continue
            group_ref = _mae(y[selected], baseline[selected])
            group_result[name] = {"n": n, "status": "SUPPORTED",
                                  "R06_mae": group_ref,
                                  "arms": {arm: {"mae": _mae(y[selected], arrays[arm]["prediction"][selected]),
                                                 "harm_vs_R06": _mae(y[selected], arrays[arm]["prediction"][selected]) - group_ref}
                                           for arm in ARMS}}
        result["groups"] = group_result
    return result


def _score_official(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    freeze = _load_freeze(source_root)
    run = ROOT / RUN_REL
    ledger = run / "fit_ledger.jsonl"
    state = _ledger_state(_read_events(ledger), freeze)
    if (len(state["starts"]) != LEDGER_CAP or len(state["saved"]) != LEDGER_CAP
            or state["failed"] or state["pending"]):
        raise PilotError("Query scoring is blocked until all six pilot fits are durable")
    prelabel_path = ROOT / REPORT_REL / "PRELABEL_VERIFICATION.json"
    prelabel = _json(prelabel_path)
    if (prelabel.get("status") != "PASS" or prelabel.get("freeze_signature") != freeze["freeze_signature"]
            or prelabel.get("all_six_models_and_predictions_durable") is not True):
        raise PilotError("Independent prelabel verification is absent or failed; query targets remain closed")
    frame = _load_frame(source_root, _json(ROOT / DESIGN_REL / "READINESS_AND_ERROR_AUDIT.json"))
    origins_data: dict[int, PreparedOrigin] = {}
    predictions: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    truths: dict[int, np.ndarray] = {}
    for origin in ORIGINS:
        # Only features and fit-only donors are rebuilt before the target read.
        prepared = _load_official_origin(source_root, frame,
                                         _json(ROOT / DESIGN_REL / "READINESS_AND_ERROR_AUDIT.json"), origin)
        origins_data[origin] = prepared
        predictions[origin] = {}
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            arrays = _load_saved_predictions(ROOT, fit_id, freeze)
            if (int(arrays["origin"].reshape(-1)[0]) != origin
                    or str(arrays["arm"].reshape(-1)[0]) != arm
                    or not np.array_equal(arrays["rows"], prepared.query_rows)
                    or not np.array_equal(arrays["record_ids"].astype(str), prepared.query_ids)
                    or not np.array_equal(arrays["cycles"], prepared.query_cycles)
                    or not np.array_equal(arrays["baseline"], prepared.query_anchor)
                    or not np.array_equal(arrays["missing"], prepared.query_missing)):
                raise PilotError(f"Prelabel prediction rows/IDs/baseline differ for {fit_id}")
            predictions[origin][arm] = arrays
        truths[origin] = _read_targets_for_rows(_contained_file(source_root, TRAIN_REL),
                                                prepared.query_rows, prepared.query_ids)

    views: dict[str, Any] = {}
    masks_67 = {
        "E67_H12": np.ones(len(origins_data[67].query_rows), dtype=bool),
        "E67_W4": (origins_data[67].query_cycles >= 68) & (origins_data[67].query_cycles <= 71),
        "E67_W5": (origins_data[67].query_cycles >= 72) & (origins_data[67].query_cycles <= 75),
        "E67_W6": (origins_data[67].query_cycles >= 76) & (origins_data[67].query_cycles <= 79),
    }
    for name, mask in masks_67.items():
        views[name] = _metric_view(origins_data[67], truths[67], predictions[67], mask,
                                   include_groups=name in ("E67_H12", "E67_W4", "E67_W5", "E67_W6"))
    mask_75 = (origins_data[75].query_cycles >= 76) & (origins_data[75].query_cycles <= 79)
    views["E75_W6"] = _metric_view(origins_data[75], truths[75], predictions[75], mask_75, include_groups=True)
    late_y = np.concatenate([truths[67][masks_67["E67_W4"]], truths[75][mask_75]])
    late_arrays: dict[str, dict[str, np.ndarray]] = {}
    for key in ("baseline", "head_prediction", "prediction"):
        for arm in ARMS:
            left = predictions[67][arm][key][masks_67["E67_W4"]]
            right = predictions[75][arm][key][mask_75]
            if arm not in late_arrays:
                late_arrays[arm] = {}
            late_arrays[arm][key] = np.concatenate([left, right])
    fake_origin = origins_data[67]
    late_mask = np.ones(len(late_y), dtype=bool)
    # The late view is an overall unique-row metric only; groups retain their
    # origin-specific fit-only boundaries and are therefore not pooled.
    late_metrics = {}
    late_base_mae = _mae(late_y, late_arrays["BAG"]["baseline"])
    late_metrics["R06"] = {"mae": late_base_mae, "spearman": _spearman(late_y, late_arrays["BAG"]["baseline"])}
    for arm in ARMS:
        pred = late_arrays[arm]["prediction"]
        late_metrics[arm] = {"mae": _mae(late_y, pred), "gain_vs_R06": late_base_mae - _mae(late_y, pred),
                             "spearman_y_vs_prediction": _spearman(late_y, pred)}
    views["LATE_UNIQUE_E67_W4_PLUS_E75_W6"] = {"n": int(len(late_y)), "metrics": late_metrics,
                                               "unique_record_ids": True, "groups": "not_applicable"}
    score = {"schema_version": 1, "status": "SCORED_AFTER_PRELABEL_PASS", "card_id": CARD_ID,
             "freeze_signature": freeze["freeze_signature"], "fit_ids": list(FIT_IDS),
             "fit_ledger_sha256": _sha_file(ledger), "fit_starts": len(state["starts"]),
             "fit_saved": len(state["saved"]), "fit_failed": len(state["failed"]),
             "prelabel_verification_sha256": _sha_file(prelabel_path),
             "query_targets_read_after_all_six_saved": True,
             "query_targets_used_for_fit": False, "test_or_sample_accessed": False,
             "classification_labels_or_probabilities_accessed": False,
             "views": views}
    path = ROOT / REPORT_REL / "SCORE_METRICS.json"
    if path.exists():
        prior = _json(path)
        if prior == score:
            return prior
        raise PilotError("Score metrics already exist with different bytes; preserve and audit")
    _atomic_json(path, score)
    return score


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little")); digest.update(relative)
        digest.update(bytes.fromhex(_sha_file(path)))
    return digest.hexdigest()


def _synthetic_origin(origin: int, *, seed: int) -> PreparedOrigin:
    rng = np.random.default_rng(seed)
    n_fit, n_query, n_bag, n_position = 180, 240, 24, 9

    def sparse_signed(rows: int, columns: int) -> sparse.csr_matrix:
        values = rng.normal(0.0, 1.0, size=(rows, columns))
        keep = rng.random((rows, columns)) < 0.18
        values[~keep] = 0.0
        return sparse.csr_matrix(values, dtype=np.float64)

    fit_bag = sparse_signed(n_fit, n_bag)
    query_bag = sparse_signed(n_query, n_bag)
    fit_position = sparse_signed(n_fit, n_position)
    query_position = sparse_signed(n_query, n_position)
    fit_baseline = np.linspace(15.0, 85.0, n_fit, dtype=np.float64)
    query_baseline = np.linspace(12.0, 88.0, n_query, dtype=np.float64)
    query_baseline += rng.normal(0.0, 0.25, size=n_query)
    query_baseline = np.clip(query_baseline, 0.0, 100.0)
    query_missing = (np.arange(n_query) % 9 == 0)
    fit_designs = {
        "BAG": head_design(fit_bag, None, fit_baseline),
        "REAL": head_design(fit_bag, fit_position, fit_baseline),
        "SHUFFLED": head_design(fit_bag, sparse_signed(n_fit, n_position), fit_baseline),
    }
    query_designs = {
        "BAG": head_design(query_bag, None, query_baseline),
        "REAL": head_design(query_bag, query_position, query_baseline),
        "SHUFFLED": head_design(query_bag, sparse_signed(n_query, n_position), query_baseline),
    }
    fit_residual = (np.asarray(fit_bag[:, 0].toarray()).reshape(-1) * 2.0
                    + rng.normal(0.0, 0.1, size=n_fit))
    query_cycles = (np.tile(np.arange(68, 80), n_query // 12) if origin == 67
                    else np.tile(np.arange(76, 80), n_query // 4))
    query_cycles = query_cycles[rng.permutation(n_query)].astype(np.int64)
    fit_cycles = rng.integers(56, min(origin, 67) + 1, size=n_fit, dtype=np.int64)
    fit_rows = rng.permutation(np.arange(10_000 + origin * 100, 10_000 + origin * 100 + n_fit)).astype(np.int64)
    query_rows = rng.permutation(np.arange(20_000 + origin * 100, 20_000 + origin * 100 + n_query)).astype(np.int64)
    fit_ids = np.asarray([f"donor-{origin}-長-{idx:04d}" for idx in range(n_fit)], dtype=str)
    query_ids = np.asarray([f"query-{origin}-α-{idx:04d}" for idx in range(n_query)], dtype=str)
    group_features = {
        "impact_load": (np.arange(n_query) % 4).astype(np.int8),
        "text_load": ((np.arange(n_query) // 3) % 4).astype(np.int8),
        "entry_friction": (np.arange(n_query) % 2).astype(np.float64),
    }
    return PreparedOrigin(
        origin=origin, fit_rows=fit_rows, fit_ids=fit_ids, fit_cycles=fit_cycles,
        fit_anchor=fit_baseline, fit_residual=fit_residual, fit_designs=fit_designs,
        query_rows=query_rows, query_ids=query_ids, query_cycles=query_cycles,
        query_anchor=query_baseline, query_missing=query_missing,
        query_designs=query_designs, group_features=group_features,
        group_edges={"impact_load": [0.0, 1.0, 2.0], "text_load": [0.0, 1.0, 2.0]},
        provenance={"origin": origin, "synthetic": True})


def _synthetic_guard_checks(origins: dict[int, PreparedOrigin], input_root: Path) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    e67 = origins[67]
    e75 = origins[75]
    try:
        _validate_query_alignment(67, 75, e67.query_rows, e67.query_ids, e67.query_cycles,
                                  e67.query_anchor, e67.query_rows, e67.query_ids,
                                  e67.query_cycles, e67.query_anchor)
    except PilotError:
        checks["wrong_origin_rejected"] = True
    else:
        raise PilotError("Synthetic negative check failed to reject the wrong origin")
    try:
        _validate_query_alignment(67, 67, e67.query_rows, e67.query_ids, e67.query_cycles,
                                  e75.query_anchor, e67.query_rows, e67.query_ids,
                                  e67.query_cycles, e67.query_anchor)
    except PilotError:
        checks["wrong_origin_baseline_rejected"] = True
    else:
        raise PilotError("Synthetic negative check failed to reject the other origin's baseline")
    donor_rows = np.arange(20, dtype=np.int64)
    donor_cycles = np.tile(np.arange(56, 60), 5).astype(np.int64)
    donor_cutoffs = (((donor_cycles - 56) // 4) * 4 + 55).astype(np.int64)
    _validate_donor_temporal(donor_rows, donor_cycles, donor_cutoffs, np.arange(30, 35), 67, 100)
    for bad_cycles, bad_cutoffs, key in (
        (np.where(np.arange(20) == 0, 68, donor_cycles), donor_cutoffs, "donor_future_cycle_rejected"),
        (donor_cycles, np.where(np.arange(20) == 0, 56, donor_cutoffs), "donor_self_cutoff_rejected"),
    ):
        try:
            _validate_donor_temporal(donor_rows, bad_cycles, bad_cutoffs, np.arange(30, 35), 67, 100)
        except PilotError:
            checks[key] = True
        else:
            raise PilotError(f"Synthetic negative check failed: {key}")

    class _VectorizerFixture:
        vocabulary_ = {"known": 0, "two words": 1, "other": 2}

        def transform(self, documents: list[str]) -> sparse.csr_matrix:
            rows = []
            for document in documents:
                rows.append([1.0 if "known" in document else 0.0,
                             1.0 if "two words" in document else 0.0,
                             1.0 if "other" in document else 0.0])
            return sparse.csr_matrix(rows, dtype=np.float64)

    zero_rows = unigram_bag(_VectorizerFixture(), ["", "unseen-oov-token"])
    if zero_rows.nnz:
        raise PilotError("Zero/OOV synthetic documents acquired unigram features")
    checks["zero_and_oov_text_produce_zero_bag"] = True
    sample_design = e67.fit_designs["REAL"]
    if not (np.any(sample_design.data < 0) and sample_design.nnz < sample_design.shape[0] * sample_design.shape[1]):
        raise PilotError("Synthetic sparse design lacks signed values or explicit zero support")
    checks["signed_and_zero_sparse_values_exercised"] = True

    bundle_path = input_root / "baseline_members.npz"
    final = e67.query_anchor
    candidate = np.clip(final + 2.0, 0.0, 100.0)
    raw = np.clip(final - 2.0, 0.0, 100.0)
    np.savez_compressed(bundle_path, baseline_prediction=final,
                        prediction=candidate, raw_candidate=raw)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        selected = _select_final_r06_baseline(bundle)
    if not np.array_equal(selected, final) or np.array_equal(selected, candidate) or np.array_equal(selected, raw):
        raise PilotError("Synthetic final-R06 selector mixed candidate/raw values into its baseline")
    checks["final_r06_member_selected_over_candidate_and_raw"] = True
    return checks


def _synthetic_manifest(origin: PreparedOrigin, raw_candidate: np.ndarray,
                        candidate_prediction: np.ndarray) -> dict[str, Any]:
    n = len(origin.query_rows)
    truth = np.clip(origin.query_anchor + np.sin(np.arange(n) / 7.0) * 3.0, 0.0, 100.0)
    truth[:8] = np.arange(8, dtype=np.float64) * 0.5
    truth[-8:] = 96.5 + np.arange(8, dtype=np.float64) * 0.5
    impact_bins = origin.group_features["impact_load"]
    text_bins = origin.group_features["text_load"]
    entry = origin.group_features["entry_friction"]
    groups = {
        "missing": origin.query_missing.tolist(),
        "target_le5": (truth <= 5.0).tolist(),
        "target_ge95": (truth >= 95.0).tolist(),
        "EF0": (entry == 0.0).tolist(),
        "EF1": (entry == 1.0).tolist(),
    }
    for name, bins in (("impact_load", impact_bins), ("text_load", text_bins)):
        for level in range(4):
            groups[f"{name}_Q{level + 1}"] = (bins == level).tolist()
    return {
        "rows": origin.query_rows.tolist(), "record_ids": origin.query_ids.tolist(),
        "cycles": origin.query_cycles.tolist(), "final_r06": origin.query_anchor.tolist(),
        "raw_candidate": raw_candidate.tolist(), "candidate_prediction": candidate_prediction.tolist(),
        "missing": origin.query_missing.tolist(), "truth": truth.tolist(), "groups": groups,
        "tail_group_counts": {"target_le5": int(np.sum(truth <= 5.0)),
                              "target_ge95": int(np.sum(truth >= 95.0))},
    }


def synthetic_e2e(temp_root: Path) -> dict[str, Any]:
    temp_root = temp_root.resolve()
    if temp_root == ROOT.resolve() or ROOT.resolve() in temp_root.parents:
        raise PilotError("Synthetic temp root must be outside the worktree")
    input_root = temp_root / "synthetic-input"
    output_root = temp_root / "synthetic-output"
    if (input_root.exists() and any(input_root.iterdir())) or (output_root.exists() and any(output_root.iterdir())):
        raise PilotError("Synthetic roots must be empty explicit test roots; refusing reuse")
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    origins = {67: _synthetic_origin(67, seed=6701), 75: _synthetic_origin(75, seed=7501)}
    guard_checks = _synthetic_guard_checks(origins, input_root)
    raw_and_candidate: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for origin in ORIGINS:
        anchor = origins[origin].query_anchor
        raw_and_candidate[origin] = (np.clip(anchor - 2.0, 0.0, 100.0),
                                     np.clip(anchor + 2.0, 0.0, 100.0))
    freeze_body = {"schema_version": 1, "card_id": CARD_ID, "synthetic_only": True,
                   "fit_ids_order": list(FIT_IDS), "accounting": {"cap": LEDGER_CAP, "retries": 0}}
    freeze = {**freeze_body, "freeze_signature": _canonical_sha(freeze_body)}
    source_sha_before = _tree_sha256(input_root)
    ledger = _reserve_ledger(output_root, input_root, freeze)
    parameters = _resolved_parameters(synthetic=True)
    fit_invocations: list[str] = []
    resume_receipt = None
    for origin in ORIGINS:
        prepared = origins[origin]
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            if not fit_invocations:
                try:
                    _fit_one(prepared, arm, fit_id=fit_id, freeze=freeze,
                             output_root=output_root, source_root=input_root, ledger=ledger,
                             parameters=parameters, interrupt_after_model=True, synthetic=True,
                             fit_invocations=fit_invocations)
                except PilotError as exc:
                    if "Synthetic interruption after durable model save" not in str(exc):
                        raise
                else:
                    raise PilotError("Synthetic interruption did not occur after durable model save")
                resume_receipt = _fit_one(prepared, arm, fit_id=fit_id, freeze=freeze,
                                          output_root=output_root, source_root=input_root, ledger=ledger,
                                          parameters=parameters, synthetic=True,
                                          fit_invocations=fit_invocations)
                if resume_receipt.get("resumed_from_durable_model_without_refit") is not True:
                    raise PilotError("Synthetic recovery did not resume without a second fit")
            else:
                _fit_one(prepared, arm, fit_id=fit_id, freeze=freeze,
                         output_root=output_root, source_root=input_root, ledger=ledger,
                         parameters=parameters, synthetic=True, fit_invocations=fit_invocations)
            gc.collect()
    events = _read_events(ledger)
    state = _ledger_state(events, freeze)
    if (len(state["starts"]) != LEDGER_CAP or len(state["saved"]) != LEDGER_CAP
            or state["failed"] or state["pending"] or len(fit_invocations) != LEDGER_CAP
            or resume_receipt is None):
        raise PilotError("Synthetic lifecycle did not reconcile to exactly six model.fit invocations")

    origin_manifest = {}
    for origin in ORIGINS:
        raw, candidate = raw_and_candidate[origin]
        origin_manifest[str(origin)] = _synthetic_manifest(origins[origin], raw, candidate)
    manifest = {"schema_version": 1,
                "freeze": {**freeze, "fit_ids_order": list(FIT_IDS)},
                "fit_invocations": len(fit_invocations),
                "resume_used_without_refit": bool(resume_receipt["resumed_from_durable_model_without_refit"]),
                "origins": origin_manifest}
    _atomic_json(input_root / "synthetic_manifest.json", manifest)
    source_sha_before_verify = _tree_sha256(input_root)
    verifier = ROOT / "scripts" / "verify_reg_position_l1_v1.py"
    checked = subprocess.run([sys.executable, str(verifier), "--phase", "synthetic",
                              "--source-root", str(input_root), "--output-root", str(output_root)],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if checked.returncode != 0:
        raise PilotError(f"Independent synthetic verifier failed: {checked.stderr[-1500:]}")
    synthetic_result = json.loads(checked.stdout)
    if _tree_sha256(input_root) != source_sha_before_verify or source_sha_before_verify == source_sha_before:
        raise PilotError("Synthetic verifier wrote to its input source root")

    tamper_root = temp_root / "tampered-output"
    shutil.copytree(output_root, tamper_root)
    tamper_id = FIT_IDS[0]
    tamper_path = _fit_artifact_paths(tamper_root, tamper_id)["predictions"]
    with np.load(tamper_path, allow_pickle=False) as archive:
        tampered = {key: archive[key] for key in archive.files}
    tampered["prediction"] = tampered["prediction"].copy()
    tampered["prediction"][0] = np.nextafter(tampered["prediction"][0], 100.0)
    _atomic_npz(tamper_path, **tampered)
    rejected = subprocess.run([sys.executable, str(verifier), "--phase", "synthetic",
                               "--source-root", str(input_root), "--output-root", str(tamper_root)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if rejected.returncode == 0:
        raise PilotError("Independent synthetic verifier accepted a tampered prediction archive")
    guard_checks["tamper_rejected_by_independent_verifier"] = True
    guard_checks["donor_future_and_self_cutoff_rejected"] = True
    guard_checks["query_cycles_and_rows_are_unsorted"] = True
    result = {"schema_version": 1, "status": "PASS_SYNTHETIC_ENTRYPOINT_E2E",
              "fit_ids": list(FIT_IDS), "fit_starts": len(state["starts"]),
              "fit_saved": len(state["saved"]), "fit_failed": len(state["failed"]),
              "fit_invocations": len(fit_invocations),
              "resumed_from_durable_model_without_refit": True,
              "source_root_sha256_before_and_after_verification": source_sha_before_verify,
              "source_root_write_guard": "PASS",
              "synthetic_independent_verification": synthetic_result,
              "guard_checks": guard_checks,
              "tail_group_counts": {key: origin_manifest["67"]["tail_group_counts"][key]
                                    for key in ("target_le5", "target_ge95")},
              "tampered_copy_rejected": True}
    _atomic_json(output_root / "synthetic_e2e_report.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_p = sub.add_parser("freeze")
    freeze_p.add_argument("--source-root", type=Path, required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--source-root", type=Path, required=True)
    fit_p.add_argument("--no-wait", action="store_true")
    score_p = sub.add_parser("score")
    score_p.add_argument("--source-root", type=Path, required=True)
    synthetic_p = sub.add_parser("synthetic-e2e")
    synthetic_p.add_argument("--temp-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_official(args.source_root)
        elif args.command == "fit":
            result = run_official_fit(args.source_root, wait_for_slot=not args.no_wait)
        elif args.command == "score":
            result = _score_official(args.source_root)
        else:
            result = synthetic_e2e(args.temp_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
