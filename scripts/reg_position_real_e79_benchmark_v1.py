"""Run one fixed REAL E79 residual benchmark with a two-unit fit ledger."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CARD_ID = "REG_POSITION_REAL_E79_EXPLORATORY_BENCHMARK_V1"
RUN_REL = Path("experiments/runs/reg_position_real_e79_benchmark_v1")
REPORT_REL = Path("reports/optimization_v3/reg_position_real_e79_benchmark_v1")
TRAIN_REL = Path("data/raw/regression/Train.csv")
TEST_REL = Path("data/raw/regression/Test.csv")
SAMPLE_REL = Path("data/raw/regression/SampleSubmission.csv")
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
TEST_SHA256 = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
SAMPLE_SHA256 = "63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32"
E79_FREEZE_REL = Path("experiments/optimization_v3/rnbr_final_e79_v1/freeze.json")
RNBR_REL = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_FINAL_E79_OBSERVED")
R06_RUNS = {
    55: Path("experiments/runs/optimization_v3_batch1/R06"),
    59: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06"),
    63: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06"),
    67: Path("experiments/runs/optimization_v3_batch1/R06"),
    71: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06"),
    75: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06"),
}
R06_CACHE_RANGES = {55: (56, 67), 59: (60, 63), 63: (64, 67),
                    67: (68, 79), 71: (72, 75), 75: (76, 79)}
TFIDF_REL = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V2_1/text_space_replays")
FEATURE_COLUMNS = ("record_id", "cycle_code", "signal_stream")
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_DONOR_ROWS = 36_010
EXPECTED_TEST_ROWS = 29_421
THREADS = 2
MAX_RSS_BYTES = 6 * 1024**3
MIN_AVAILABLE_BEFORE_FIT = 6 * 1024**3
MIN_AVAILABLE_DURING_FIT = 2 * 1024**3
FIT_UNITS = ("WORD_TFIDF_FULL_TRAIN", "REAL_POSITION_RESIDUAL_L1_E79")
CODE_RELS = (
    Path("scripts/reg_position_real_e79_benchmark_v1.py"),
    Path("scripts/verify_reg_position_real_e79_benchmark_v1.py"),
    Path("src/models/classification_models.py"),
    Path("src/models/optimization_v3_positional_text_pilot_v1.py"),
    Path("src/models/reg_position_l1_v1.py"),
    Path("src/models/regression_research_tabular.py"),
    REPORT_REL / "BENCHMARK_AUTHORIZATION.md",
)
FINAL_UPLOAD_LABEL = "Samuel REG Position L1 E79 Exploratory Benchmark"


class BenchmarkError(RuntimeError):
    """Raised when a frozen input, boundary, resource gate, or artifact fails."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(str).tolist() if value.dtype.kind in "OUS" else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(_canonical({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(_canonical(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _rnbr_sha_array(value: Any, dtype: str | None = None) -> str:
    """Match the existing RNBR E79 receipt's raw-array digest contract."""
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "OUS":
        digest = hashlib.sha256()
        for item in array.reshape(-1):
            encoded = str(item).encode("utf-8", errors="surrogatepass")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return digest.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _sha_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return _canonical_sha({
        "shape": list(matrix.shape),
        "data_sha256": _sha_array(matrix.data),
        "indices_sha256": _sha_array(matrix.indices),
        "indptr_sha256": _sha_array(matrix.indptr),
    })


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected JSON object at {path}")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise BenchmarkError(f"Refusing to overwrite different bytes: {path}")
        return
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        raise BenchmarkError(f"Stale staging file needs review: {staging}")
    with staging.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BenchmarkError(f"Refusing to overwrite a fit state: {path}")
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".staging", dir=path.parent)
    os.close(fd)
    staging = Path(name)
    try:
        joblib.dump(value, staging, compress=0)
        with staging.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _save_npz_once(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):
                raise BenchmarkError(f"Existing inference artifact schema differs: {path}")
            for key, value in arrays.items():
                if not np.array_equal(old[key], np.asarray(value)):
                    raise BenchmarkError(f"Existing inference artifact differs at {key}")
        return
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".staging", dir=path.parent)
    os.close(fd)
    staging = Path(name)
    try:
        with staging.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def _contained(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise BenchmarkError(f"Missing or out-of-root input: {relative}")
    return path


def _normalize_tfidf_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(params)
    if "ngram_range" in result:
        result["ngram_range"] = list(result["ngram_range"])
    if "dtype" in result:
        result["dtype"] = np.dtype(result["dtype"]).name
    return _jsonable(result)


def _resolved_tfidf_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_tfidf_parameters(TfidfVectorizer(**dict(params)).get_params(deep=False))


def _resolved_head_parameters() -> dict[str, Any]:
    from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS

    model = LGBMRegressor(**{**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"]})
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in sorted(model.get_params(deep=False).items())
    }


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head(worktree_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree_root,
                                   text=True, encoding="utf-8").strip()


def _run_root(worktree_root: Path, output_root: Path) -> Path:
    expected = (worktree_root.resolve() / RUN_REL).resolve()
    output = output_root.resolve()
    if output != expected:
        raise BenchmarkError(f"Output root must be exactly {expected}")
    return output


def _check_roots(source_root: Path, worktree_root: Path) -> None:
    source, worktree = source_root.resolve(), worktree_root.resolve()
    if source == worktree or source in worktree.parents or worktree in source.parents:
        raise BenchmarkError("Main source root and isolated worktree must be disjoint")
    if ROOT.resolve() != worktree:
        raise BenchmarkError("Worktree root does not match this script's checkout")


def _read_train_features(source_root: Path) -> pd.DataFrame:
    path = _contained(source_root, TRAIN_REL)
    if _sha_file(path) != TRAIN_SHA256:
        raise BenchmarkError("Official Regression Train hash differs from authorization")
    frame = pd.read_csv(path, usecols=list(FEATURE_COLUMNS), dtype={"record_id": "string"})
    frame = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    cycle_values = pd.to_numeric(frame.cycle_code, errors="coerce").to_numpy()
    if (len(frame) != EXPECTED_TRAIN_ROWS or len(set(ids.tolist())) != EXPECTED_TRAIN_ROWS
            or not np.isfinite(cycle_values).all() or np.any(cycle_values != np.floor(cycle_values))
            or np.any((cycle_values < 47) | (cycle_values > 79))):
        raise BenchmarkError("Train feature rows, IDs or cycles failed integrity checks")
    frame["cycle_code"] = cycle_values.astype(np.int64)
    if "threat_rank" in frame.columns or "artifact_signal" in frame.columns:
        raise BenchmarkError("Target column entered the target-free feature frame")
    return frame


def _read_targets_for_rows(
    path: Path, rows: np.ndarray, expected_ids: np.ndarray, expected_cycles: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    ids = np.asarray(expected_ids, dtype=str)
    cycles = np.asarray(expected_cycles, dtype=np.int64)
    if (rows.ndim != 1 or ids.shape != rows.shape or cycles.shape != rows.shape
            or len(set(rows.tolist())) != len(rows)):
        raise BenchmarkError("Donor rows, IDs and cycles must be unique aligned arrays")
    wanted = {int(row): i for i, row in enumerate(rows)}
    targets = np.full(len(rows), np.nan, dtype=np.float64)
    seen: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)):
            raise BenchmarkError("Official Train has duplicate column names")
        try:
            id_col = header.index("record_id")
            cycle_col = header.index("cycle_code")
            text_col = header.index("signal_stream")
            target_col = header.index("threat_rank")
        except ValueError as exc:
            raise BenchmarkError("Official Train lacks a donor source column") from exc
        for row_pos, row in enumerate(reader):
            slot = wanted.get(row_pos)
            if slot is None:
                continue
            if (len(row) != len(header) or row[id_col] != ids[slot]
                    or int(row[cycle_col]) != int(cycles[slot])
                    or not row[text_col].strip() or not 56 <= int(row[cycle_col]) <= 79):
                raise BenchmarkError("Donor target read crossed row/ID/cycle/text boundaries")
            value = float(row[target_col])
            if not np.isfinite(value) or not 0.0 <= value <= 100.0:
                raise BenchmarkError("Donor target is not finite in [0,100]")
            targets[slot] = value
            seen.append(row_pos)
    if seen != rows.tolist() or not np.isfinite(targets).all():
        raise BenchmarkError("Not every authorized donor target was read exactly once")
    return targets


def _signature_valid(value: Mapping[str, Any], key: str = "freeze_signature") -> bool:
    signature = value.get(key)
    body = {name: item for name, item in value.items() if name != key}
    return isinstance(signature, str) and _canonical_sha(body) == signature


def _pilot_tfidf_recipe(source_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    from src.models.classification_models import _TFIDF_PARAMS

    resolved = _normalize_tfidf_parameters(_TFIDF_PARAMS)
    sklearn_resolved = _resolved_tfidf_parameters(_TFIDF_PARAMS)
    receipt_hashes: dict[str, str] = {}
    for origin in (67, 75):
        folder = TFIDF_REL / f"E{origin}"
        receipt_path = _contained(source_root, folder / "receipt.json")
        state_path = _contained(source_root, folder / "text_state.json")
        vector_path = _contained(source_root, folder / "vectorizer.joblib")
        receipt = _read_json(receipt_path)
        state = _read_json(state_path)
        body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != _canonical_sha(body)
                or receipt.get("origin_cycle") != origin
                or receipt.get("fit_kind") != "TARGET_FREE_TFIDF_REPLAY"
                or receipt.get("query_targets_read") is not False
                or receipt.get("classification_data_read") is not False
                or receipt.get("fit_transform_calls") != 1
                or receipt.get("vectorizer_sha256") != _sha_file(vector_path)
                or receipt.get("text_state_sha256") != _sha_file(state_path)
                or state.get("audit") != receipt.get("text_audit")):
            raise BenchmarkError(f"E{origin} standalone pilot vectorizer receipt failed")
        if _normalize_tfidf_parameters(receipt["text_audit"]["tfidf_parameters"]) != resolved:
            raise BenchmarkError(f"E{origin} pilot vectorizer parameters differ from source")
        vectorizer = joblib.load(vector_path)
        actual = _normalize_tfidf_parameters(vectorizer.get_params(deep=False))
        if (not isinstance(vectorizer, TfidfVectorizer)
                or any(actual.get(key) != value for key, value in resolved.items())
                or vectorizer.norm != "l2"):
            raise BenchmarkError(f"E{origin} saved standalone vectorizer state failed")
        receipt_hashes[str(origin)] = _sha_file(receipt_path)
    return resolved, sklearn_resolved, receipt_hashes


def _r06_oof(
    source_root: Path, source_freeze: Mapping[str, Any], frame: pd.DataFrame,
) -> tuple[dict[int, dict[int, float]], dict[str, str]]:
    frozen = source_freeze.get("input_sha256", {})
    if not isinstance(frozen, dict):
        raise BenchmarkError("Existing E79 source freeze lacks R06 input hashes")
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    caches: dict[int, dict[int, float]] = {}
    hashes: dict[str, str] = {}
    for cutoff, folder in R06_RUNS.items():
        npz_rel, side_rel = folder / f"R06_fit{cutoff}.npz", folder / f"R06_fit{cutoff}.json"
        npz_path, side_path = _contained(source_root, npz_rel), _contained(source_root, side_rel)
        for rel, path in ((npz_rel.as_posix(), npz_path), (side_rel.as_posix(), side_path)):
            actual = _sha_file(path)
            if frozen.get(rel) != actual:
                raise BenchmarkError(f"R06 source differs from E79 freeze: {rel}")
            hashes[rel] = actual
        receipt = _read_json(side_path)
        identity = receipt.get("identity")
        if not isinstance(identity, dict):
            raise BenchmarkError(f"R06 cutoff {cutoff} receipt has no identity")
        integrity = {key: item for key, item in receipt.items() if key != "integrity_sha256"}
        if (receipt.get("schema_version") != 1
                or receipt.get("integrity_sha256") != _canonical_sha(integrity)
                or receipt.get("identity_sha256") != _canonical_sha(identity)
                or identity.get("card_id") != "R06" or identity.get("cutoff") != cutoff
                or receipt.get("npz_sha256") != _sha_file(npz_path)):
            raise BenchmarkError(f"R06 cutoff {cutoff} receipt integrity failed")
        with np.load(npz_path, allow_pickle=False) as archive:
            if not {"score_rows", "score_record_ids", "prediction"}.issubset(archive.files):
                raise BenchmarkError(f"R06 cutoff {cutoff} lacks final mapped predictions")
            rows = np.asarray(archive["score_rows"], dtype=np.int64)
            record_ids = np.asarray(archive["score_record_ids"], dtype=str)
            prediction = np.asarray(archive["prediction"], dtype=np.float64)
            score_cycles = (np.asarray(archive["score_cycles"], dtype=np.int64)
                            if "score_cycles" in archive.files else None)
        lo, hi = R06_CACHE_RANGES[cutoff]
        valid = (
            rows.ndim == 1 and len(rows) == len(set(rows.tolist()))
            and record_ids.shape == rows.shape and prediction.shape == rows.shape
            and np.array_equal(rows, np.asarray(identity.get("score_rows"), dtype=np.int64))
            and not np.any((rows < 0) | (rows >= len(frame)))
            and np.array_equal(record_ids, ids[rows])
            and np.all((cycles[rows] >= lo) & (cycles[rows] <= hi))
            and (score_cycles is None or np.array_equal(score_cycles, cycles[rows]))
            and np.isfinite(prediction).all()
            and not np.any((prediction < 0.0) | (prediction > 100.0))
        )
        if not valid:
            raise BenchmarkError(f"R06 cutoff {cutoff} row/ID/cycle/prediction failed")
        caches[cutoff] = {int(row): float(value) for row, value in zip(rows, prediction)}
    return caches, hashes


def _compare_rnbr_bank(
    source_root: Path, source_freeze: Mapping[str, Any],
    rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
    targets: np.ndarray, oof: np.ndarray, cutoffs: np.ndarray, residuals: np.ndarray,
) -> dict[str, Any]:
    receipt_path = _contained(source_root, RNBR_REL / "fit_receipt.json")
    bundle_path = _contained(source_root, RNBR_REL / "e79_bundle.npz")
    matrix_path = _contained(source_root, RNBR_REL / "donor_tfidf_csr.npz")
    receipt = _read_json(receipt_path)
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != _canonical_sha(body)
            or receipt.get("fit_id") != "RNBR_FINAL_E79_OBSERVED"
            or receipt.get("status") != "FIT_OUTPUT_DURABLE"
            or receipt.get("freeze_signature") != source_freeze.get("freeze_signature")
            or receipt.get("bundle_sha256") != _sha_file(bundle_path)
            or receipt.get("bank_matrix_file_sha256") != _sha_file(matrix_path)):
        raise BenchmarkError("Audited RNBR E79 donor bank receipt/hash failed")
    with np.load(bundle_path, allow_pickle=False) as archive:
        checks = {
            "donor_rows": np.array_equal(archive["donor_rows"], rows),
            "donor_ids": np.array_equal(archive["donor_ids"].astype(str), ids),
            "donor_cycles": np.array_equal(archive["donor_cycles"], cycles),
            "donor_targets": np.array_equal(archive["donor_targets"], targets),
            "donor_final_r06_prediction": np.array_equal(archive["donor_r06_oof"], oof),
            "donor_oof_cutoffs": np.array_equal(archive["donor_oof_cutoffs"], cutoffs),
            "donor_oof_ids": np.array_equal(archive["donor_oof_ids"].astype(str), ids),
            "donor_residuals": np.array_equal(archive["donor_residuals"], residuals),
        }
    observed_hashes = {
        "donor_rows_sha256": _rnbr_sha_array(rows, "<i8"),
        "donor_ids_sha256": _rnbr_sha_array(ids),
        "donor_targets_sha256": _rnbr_sha_array(targets, "<f8"),
        "donor_r06_oof_sha256": _rnbr_sha_array(oof, "<f8"),
        "donor_residual_sha256": _rnbr_sha_array(residuals, "<f8"),
    }
    observed_cutoff_counts = {
        str(int(k)): int(v) for k, v in zip(*np.unique(cutoffs, return_counts=True))
    }
    if (not all(checks.values()) or any(receipt.get(key) != value for key, value in observed_hashes.items())
            or receipt.get("donor_oof_cutoffs") != observed_cutoff_counts
            or receipt.get("donor_rows") != EXPECTED_DONOR_ROWS):
        raise BenchmarkError(f"RNBR E79 donor bank does not exactly match verified sources: {checks}")
    return {
        "status": "EXACT_MATCH",
        "fit_receipt_sha256": _sha_file(receipt_path),
        "bundle_sha256": _sha_file(bundle_path),
        "matrix_sha256": _sha_file(matrix_path),
        "member_used_for_final_r06_oof": "prediction",
        "excluded_members": ["baseline_prediction", "raw_candidate"],
        "checks": checks,
        "verified_hashes": observed_hashes,
    }


def _collect_inputs(source_root: Path) -> dict[str, Any]:
    from src.models.classification_models import _text_values

    source_root = source_root.resolve()
    source_freeze_path = _contained(source_root, E79_FREEZE_REL)
    source_freeze = _read_json(source_freeze_path)
    if (not _signature_valid(source_freeze)
            or source_freeze.get("protocol_id") != "RNBR_FINAL_E79_OBSERVED_TEST_BENCHMARK_V1"
            or source_freeze.get("official_train", {}).get("sha256") != TRAIN_SHA256):
        raise BenchmarkError("Existing official E79 source freeze failed integrity")
    frame = _read_train_features(source_root)
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    documents, missing = _text_values(frame.signal_stream)
    donor_rows = np.flatnonzero((cycles >= 56) & (cycles <= 79) & ~missing).astype(np.int64)
    if len(donor_rows) != EXPECTED_DONOR_ROWS:
        raise BenchmarkError(f"Present-stream donor count differs from authorized expected rows: {len(donor_rows)}")
    donor_ids, donor_cycles = ids[donor_rows], cycles[donor_rows]
    targets = _read_targets_for_rows(
        _contained(source_root, TRAIN_REL), donor_rows, donor_ids, donor_cycles,
    )
    caches, oof_hashes = _r06_oof(source_root, source_freeze, frame)
    cutoffs = ((donor_cycles - 56) // 4 * 4 + 55).astype(np.int64)
    if np.any(cutoffs >= donor_cycles):
        raise BenchmarkError("Donor OOF source cutoff is not strictly before its cycle")
    oof = np.empty(len(donor_rows), dtype=np.float64)
    for cutoff in np.unique(cutoffs):
        ix = np.flatnonzero(cutoffs == cutoff)
        cache = caches[int(cutoff)]
        for i in ix:
            row = int(donor_rows[i])
            if row not in cache:
                raise BenchmarkError(f"R06 final OOF cutoff {cutoff} misses donor row {row}")
            oof[i] = cache[row]
    residuals = targets - oof
    if not np.isfinite(residuals).all():
        raise BenchmarkError("Donor residual vector is not finite")
    tfidf_params, sklearn_tfidf_params, pilot_receipts = _pilot_tfidf_recipe(source_root)
    rnbr_comparison = _compare_rnbr_bank(
        source_root, source_freeze, donor_rows, donor_ids, donor_cycles,
        targets, oof, cutoffs, residuals,
    )
    hashes: dict[str, str] = {
        TRAIN_REL.as_posix(): _sha_file(_contained(source_root, TRAIN_REL)),
        E79_FREEZE_REL.as_posix(): _sha_file(source_freeze_path),
        **oof_hashes,
    }
    for relative in (
        RNBR_REL / "fit_receipt.json", RNBR_REL / "e79_bundle.npz",
        RNBR_REL / "donor_tfidf_csr.npz",
    ):
        hashes[relative.as_posix()] = _sha_file(_contained(source_root, relative))
    for origin in (67, 75):
        for name in ("receipt.json", "text_state.json", "vectorizer.joblib"):
            rel = TFIDF_REL / f"E{origin}" / name
            hashes[rel.as_posix()] = _sha_file(_contained(source_root, rel))
    if len(set(donor_rows.tolist())) != EXPECTED_DONOR_ROWS:
        raise BenchmarkError("Donor positions are not unique")
    return {
        "source_root": str(source_root),
        "frame": frame,
        "ids": ids,
        "cycles": cycles,
        "documents": documents,
        "donor_rows": donor_rows,
        "donor_ids": donor_ids,
        "donor_cycles": donor_cycles,
        "donor_targets": targets,
        "donor_r06_oof": oof,
        "donor_cutoffs": cutoffs,
        "donor_residuals": residuals,
        "source_freeze": source_freeze,
        "input_hashes": dict(sorted(hashes.items())),
        "tfidf_parameters": tfidf_params,
        "sklearn_tfidf_parameters": sklearn_tfidf_params,
        "pilot_tfidf_receipts": pilot_receipts,
        "rnbr_comparison": rnbr_comparison,
    }


def _worktree_hashes(worktree_root: Path) -> dict[str, str]:
    return {rel.as_posix(): _sha_file(_contained(worktree_root, rel)) for rel in CODE_RELS}


def _preflight_payload(
    source_root: Path, worktree_root: Path, output_root: Path, inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "card_id": CARD_ID,
        "status": "PASS_ZERO_FIT_PREPARATION",
        "source_root": str(source_root.resolve()),
        "worktree_root": str(worktree_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_freeze_signature": inputs["source_freeze"]["freeze_signature"],
        "input_hashes": inputs["input_hashes"],
        "worktree_source_hashes": _worktree_hashes(worktree_root),
        "official_train": {
            "rows": EXPECTED_TRAIN_ROWS,
            "sha256": inputs["input_hashes"][TRAIN_REL.as_posix()],
            "cycle_bounds": [47, 79],
            "vectorizer_fit_rows": EXPECTED_TRAIN_ROWS,
        },
        "pilot_tfidf": {
            "resolved_parameters": inputs["tfidf_parameters"],
            "sklearn_resolved_parameters": inputs["sklearn_tfidf_parameters"],
            "receipt_sha256": inputs["pilot_tfidf_receipts"],
        },
        "head_parameters": _resolved_head_parameters(),
        "donor_bank": {
            "rows": len(inputs["donor_rows"]),
            "cycles": [int(inputs["donor_cycles"].min()), int(inputs["donor_cycles"].max())],
            "cycle_counts": {
                str(int(k)): int(v) for k, v in
                zip(*np.unique(inputs["donor_cycles"], return_counts=True))
            },
            "cutoff_counts": {
                str(int(k)): int(v) for k, v in
                zip(*np.unique(inputs["donor_cutoffs"], return_counts=True))
            },
            "rows_sha256": _sha_array(inputs["donor_rows"], "<i8"),
            "ids_sha256": _sha_array(inputs["donor_ids"]),
            "targets_sha256": _sha_array(inputs["donor_targets"], "<f8"),
            "final_r06_oof_sha256": _sha_array(inputs["donor_r06_oof"], "<f8"),
            "cutoffs_sha256": _sha_array(inputs["donor_cutoffs"], "<i8"),
            "residuals_sha256": _sha_array(inputs["donor_residuals"], "<f8"),
            "rnbr_e79_comparison": inputs["rnbr_comparison"],
        },
        "accounting": {
            "learned_units_total": 2, "units": list(FIT_UNITS),
            "fit_starts": 0, "retries": 0, "shared_main_ledger_write": False,
        },
        "boundaries": {
            "test_features_opened": False, "sample_submission_opened": False,
            "test_labels_read": False, "classification_data_read": False,
            "other_model_fits": 0, "new_predictions": 0, "kaggle_accessed": False,
        },
    }


def run_preflight(source_root: Path, worktree_root: Path, output_root: Path) -> dict[str, Any]:
    source_root, worktree_root = source_root.resolve(), worktree_root.resolve()
    _check_roots(source_root, worktree_root)
    output_root = _run_root(worktree_root, output_root)
    auth = _contained(worktree_root, REPORT_REL / "BENCHMARK_AUTHORIZATION.md")
    auth_text = auth.read_text(encoding="utf-8")
    if CARD_ID not in auth_text or "EXPLORATORY" not in auth_text or "authorized" not in auth_text.lower():
        raise BenchmarkError("Explicit benchmark authorization file is absent or incomplete")
    inputs = _collect_inputs(source_root)
    payload = _preflight_payload(source_root, worktree_root, output_root, inputs)
    _atomic_json(output_root / "preflight.json", payload)
    if (output_root / "fit_ledger.jsonl").exists():
        raise BenchmarkError("Zero-fit preflight found an existing fit ledger")
    return payload


def _load_preflight(source_root: Path, worktree_root: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = _collect_inputs(source_root)
    expected = _preflight_payload(source_root, worktree_root, output_root, inputs)
    path = output_root / "preflight.json"
    if not path.is_file() or _read_json(path) != expected:
        raise BenchmarkError("Zero-fit preflight is missing or differs from current frozen inputs")
    return expected, inputs


def _process_identity(proc: psutil.Process) -> tuple[str, bool, float]:
    try:
        command = proc.cmdline()
        script = next((Path(x.strip('"')).name for x in command if x.lower().endswith(".py")), None)
        identity = script or ("interactive-python" if len(command) <= 1 else "other-python")
        joined = " ".join(command).lower()
        known = any(x in joined for x in (
            "reg_position_real_e79_benchmark_v1", "reg_position_l1_v1",
            "optimization_v3_", "lightgbm",
        ))
        cpu = proc.cpu_times()
        return identity, known, float(cpu.user + cpu.system)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return "unavailable-process-metadata", False, 0.0


def _active_trainers() -> list[dict[str, Any]]:
    current = os.getpid()
    candidates = []
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        if proc.info.get("pid") == current or "python" not in str(proc.info.get("name", "")).lower():
            continue
        ident, known, cpu = _process_identity(proc)
        candidates.append((proc, ident, known, cpu))
    time.sleep(1.0)
    active = []
    for proc, ident, known, before in candidates:
        try:
            now = proc.cpu_times()
            delta = float(now.user + now.system) - before
            if known or delta >= 0.5:
                active.append({"pid": int(proc.pid), "identity": ident,
                               "known_fit_command": known, "cpu_seconds_1s": round(delta, 3)})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return active


def _resource_preflight(*, synthetic: bool = False) -> dict[str, Any]:
    process = psutil.Process()
    available = int(psutil.virtual_memory().available)
    rss = int(process.memory_info().rss)
    active = [] if synthetic else _active_trainers()
    snapshot = {
        "available_ram_bytes": available, "process_rss_bytes": rss,
        "minimum_available_before_fit_bytes": MIN_AVAILABLE_BEFORE_FIT,
        "minimum_available_during_fit_bytes": MIN_AVAILABLE_DURING_FIT,
        "max_process_rss_bytes": MAX_RSS_BYTES, "thread_limit": THREADS,
        "active_training_processes": active,
    }
    if not synthetic and (available < MIN_AVAILABLE_BEFORE_FIT or rss >= MAX_RSS_BYTES or active):
        raise BenchmarkError(f"Compute/resource gate failed before fit start: {snapshot}")
    return snapshot


def _resource_after_fit(*, synthetic: bool = False) -> dict[str, int]:
    available = int(psutil.virtual_memory().available)
    rss = int(psutil.Process().memory_info().rss)
    snapshot = {"available_ram_bytes": available, "process_rss_bytes": rss}
    if not synthetic and (available < MIN_AVAILABLE_DURING_FIT or rss > MAX_RSS_BYTES):
        raise BenchmarkError(f"Compute/resource gate failed after fit: {snapshot}")
    return snapshot


def _append_event(output_root: Path, event: Mapping[str, Any]) -> None:
    ledger = (output_root.resolve() / "fit_ledger.jsonl").resolve()
    if output_root.resolve() not in ledger.parents:
        raise BenchmarkError("Fit ledger escaped the isolated output root")
    if ledger.exists():
        lines = ledger.read_text(encoding="utf-8").splitlines()
        if lines and not ledger.read_bytes().endswith(b"\n"):
            raise BenchmarkError("Fit ledger has an incomplete trailing line")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("ab") as stream:
        stream.write(_canonical(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _events(output_root: Path) -> list[dict[str, Any]]:
    ledger = output_root / "fit_ledger.jsonl"
    if not ledger.is_file():
        return []
    result = []
    for line_no, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except Exception as exc:
            raise BenchmarkError(f"Invalid fit ledger line {line_no}") from exc
        if not isinstance(value, dict):
            raise BenchmarkError(f"Fit ledger line {line_no} is not an object")
        result.append(value)
    return result


def _validate_ledger(output_root: Path) -> dict[str, Any]:
    events = _events(output_root)
    reservations = [e for e in events if e.get("event") == "FIT_BUDGET_RESERVED"]
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    start_ids = [e.get("fit_id") for e in starts]
    terminal_ids = [e.get("fit_id") for e in terminals]
    saved = {e.get("fit_id") for e in terminals if e.get("event") == "FIT_SAVED"}
    if (len(reservations) != 1 or reservations[0].get("fit_ids") != list(FIT_UNITS)
            or len(start_ids) != len(set(start_ids)) or len(terminal_ids) != len(set(terminal_ids))
            or set(start_ids) - set(FIT_UNITS) or set(terminal_ids) - set(start_ids)
            or len(start_ids) > 2 or any(e.get("event") == "FIT_FAILED" for e in terminals)
            or saved != set(start_ids)):
        raise BenchmarkError("Two-unit fit ledger is invalid, failed, or contains a retry")
    return {
        "fit_starts": len(start_ids), "fit_saved": len(saved),
        "fit_ids": start_ids,
        "ledger_sha256": _sha_file(output_root / "fit_ledger.jsonl"),
        "retries": 0,
    }


def _unit_paths(output_root: Path, fit_id: str) -> tuple[Path, Path]:
    return (
        output_root / "states" / f"{fit_id}.joblib",
        output_root / "receipts" / f"{fit_id}.json",
    )


def _fit_unit(
    output_root: Path, fit_id: str, input_hashes: Mapping[str, Any],
    fit_fn: Callable[[], Any], validate: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
    fit_metadata: Mapping[str, Any], *, synthetic: bool = False,
    call_log: list[str] | None = None,
) -> tuple[Any, dict[str, Any], bool]:
    if fit_id not in FIT_UNITS:
        raise BenchmarkError(f"Unapproved learned unit: {fit_id}")
    state_path, receipt_path = _unit_paths(output_root, fit_id)
    relevant = [e for e in _events(output_root) if e.get("fit_id") == fit_id]
    starts = [e for e in relevant if e.get("event") == "FIT_STARTED"]
    saved = [e for e in relevant if e.get("event") == "FIT_SAVED"]
    failed = [e for e in relevant if e.get("event") == "FIT_FAILED"]

    def load_existing() -> tuple[Any, dict[str, Any]]:
        if not state_path.is_file() or not receipt_path.is_file():
            raise BenchmarkError(f"Fit {fit_id} has no complete durable artifact/receipt")
        receipt = _read_json(receipt_path)
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != _canonical_sha(body)
                or receipt.get("fit_id") != fit_id
                or receipt.get("input_sha256") != _jsonable(input_hashes)
                or receipt.get("artifact_sha256") != _sha_file(state_path)):
            raise BenchmarkError(f"Fit {fit_id} receipt or model hash failed")
        payload = joblib.load(state_path)
        if _jsonable(validate(payload, receipt)) != receipt.get("output_audit"):
            raise BenchmarkError(f"Fit {fit_id} saved state audit failed")
        return payload, receipt

    if len(starts) > 1 or len(saved) > 1 or len(failed) > 1 or (saved and failed):
        raise BenchmarkError(f"Fit {fit_id} has duplicate ledger events")
    if saved:
        payload, receipt = load_existing()
        if saved[0].get("receipt_sha256") != receipt["receipt_sha256"]:
            raise BenchmarkError(f"Fit {fit_id} ledger receipt differs")
        return payload, receipt, True
    if failed:
        raise BenchmarkError(f"Fit {fit_id} failed previously; its slot is consumed")
    if starts:
        if state_path.exists() and receipt_path.exists():
            payload, receipt = load_existing()
            _append_event(output_root, {
                "at_utc": _utc(), "event": "FIT_SAVED", "fit_id": fit_id,
                "receipt_sha256": receipt["receipt_sha256"],
                "recovered_after_durable_state": True,
            })
            return payload, receipt, True
        raise BenchmarkError(f"Fit {fit_id} started without a durable state; no retry")
    if state_path.exists() or receipt_path.exists():
        raise BenchmarkError(f"Fit {fit_id} has artifacts without a start event")
    resources = _resource_preflight(synthetic=synthetic)
    _append_event(output_root, {
        "at_utc": _utc(), "event": "FIT_STARTED", "fit_id": fit_id,
        "fit_consumed_at_start": True, "input_sha256": _jsonable(input_hashes),
        "resource_preflight": resources,
    })
    started = time.perf_counter()
    try:
        if call_log is not None:
            call_log.append(fit_id)
        with threadpool_limits(limits=THREADS):
            payload = fit_fn()
        metadata = _jsonable(fit_metadata)
        output_audit = _jsonable(validate(payload, metadata))
        _atomic_joblib(state_path, payload)
        receipt = {
            "schema_version": 1, "fit_id": fit_id, "status": "FIT_OUTPUT_DURABLE",
            "input_sha256": _jsonable(input_hashes), **metadata,
            "fit_elapsed_seconds": round(time.perf_counter() - started, 6),
            "resource_preflight": resources,
            "resource_after_fit": _resource_after_fit(synthetic=synthetic),
            "artifact_sha256": _sha_file(state_path),
            "output_audit": output_audit,
            "shared_main_ledger_written": False,
        }
        receipt["receipt_sha256"] = _canonical_sha(receipt)
        _atomic_json(receipt_path, receipt)
        _append_event(output_root, {
            "at_utc": _utc(), "event": "FIT_SAVED", "fit_id": fit_id,
            "artifact_sha256": receipt["artifact_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
        })
        return payload, receipt, False
    except Exception as exc:
        _append_event(output_root, {
            "at_utc": _utc(), "event": "FIT_FAILED", "fit_id": fit_id,
            "fit_consumed_at_start": True, "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        })
        raise


def _vectorizer_audit(value: Any, fit_metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, TfidfVectorizer) or not hasattr(value, "vocabulary_"):
        raise BenchmarkError("Saved target-free vectorizer is not fitted")
    actual = _normalize_tfidf_parameters(value.get_params(deep=False))
    recipe_parameters = fit_metadata.get("parameters")
    if (fit_metadata.get("sklearn_resolved_parameters") != actual
            or any(actual.get(key) != item for key, item in recipe_parameters.items())
            or value.norm != "l2"):
        raise BenchmarkError("Saved target-free vectorizer parameters differ")
    vocab = sorted((str(term), int(index)) for term, index in value.vocabulary_.items())
    return {
        "type": "sklearn.TfidfVectorizer",
        "fit_rows": int(fit_metadata["fit_rows"]),
        "vocabulary_size": len(vocab),
        "vocabulary_sha256": _canonical_sha(vocab),
        "idf_sha256": _sha_array(value.idf_, "<f8"),
        "parameters": recipe_parameters,
        "sklearn_resolved_parameters": actual,
    }


def _head_audit(value: Any, fit_metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, LGBMRegressor) or not hasattr(value, "booster_"):
        raise BenchmarkError("Saved REAL residual head is not fitted")
    actual = {
        key: (item.item() if isinstance(item, np.generic) else item)
        for key, item in sorted(value.get_params(deep=False).items())
    }
    if actual != fit_metadata.get("parameters"):
        raise BenchmarkError("Saved REAL residual head parameters differ")
    return {
        "type": "lightgbm.LGBMRegressor",
        "n_features_in": int(value.n_features_in_),
        "n_estimators": int(value.n_estimators),
        "parameters": actual,
    }


def _build_design(vectorizer: TfidfVectorizer, documents: list[str],
                  baseline: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
    from src.models.reg_position_l1_v1 import head_design, unigram_bag

    bag = unigram_bag(vectorizer, documents)
    position, _ = build_position_block(documents, vectorizer, shuffled=False)
    design = head_design(bag, position, np.asarray(baseline, dtype=np.float64))
    if design.shape[1] != bag.shape[1] + position.shape[1] + 1:
        raise BenchmarkError("REAL feature design is not [B,P,C]")
    return bag, position, design


def _fit_vectorizer(documents: list[str], params: Mapping[str, Any]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(**dict(params))
    vectorizer.fit(documents)
    return vectorizer


def run_fit(source_root: Path, worktree_root: Path, output_root: Path) -> dict[str, Any]:
    from src.models.classification_models import _TFIDF_PARAMS
    from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS

    source_root, worktree_root = source_root.resolve(), worktree_root.resolve()
    _check_roots(source_root, worktree_root)
    output_root = _run_root(worktree_root, output_root)
    preflight, inputs = _load_preflight(source_root, worktree_root, output_root)
    trained_path = output_root / "trained_freeze.json"
    if trained_path.exists():
        return _verify_trained_freeze(source_root, worktree_root, output_root)
    if not (output_root / "fit_ledger.jsonl").exists():
        _append_event(output_root, {
            "at_utc": _utc(), "event": "FIT_BUDGET_RESERVED",
            "card_id": CARD_ID, "fit_ids": list(FIT_UNITS),
            "fit_cap": 2, "retries": 0, "shared_main_ledger_written": False,
        })
    elif len([e for e in _events(output_root) if e.get("event") == "FIT_BUDGET_RESERVED"]) != 1:
        raise BenchmarkError("Fit ledger budget reservation is missing or duplicated")

    vectorizer_inputs = {
        "train_sha256": preflight["official_train"]["sha256"],
        "all_train_documents_sha256": _canonical_sha(inputs["documents"]),
        "tfidf_parameters_sha256": _canonical_sha(preflight["pilot_tfidf"]["resolved_parameters"]),
        "sklearn_tfidf_parameters_sha256": _canonical_sha(preflight["pilot_tfidf"]["sklearn_resolved_parameters"]),
        "frozen_source_inputs_sha256": _canonical_sha(preflight["input_hashes"]),
    }
    vectorizer_meta = {
        "fit_rows": EXPECTED_TRAIN_ROWS,
        "parameters": preflight["pilot_tfidf"]["resolved_parameters"],
        "sklearn_resolved_parameters": preflight["pilot_tfidf"]["sklearn_resolved_parameters"],
    }
    vectorizer, vector_receipt, _ = _fit_unit(
        output_root, FIT_UNITS[0], vectorizer_inputs,
        lambda: _fit_vectorizer(inputs["documents"], _TFIDF_PARAMS),
        _vectorizer_audit, vectorizer_meta,
    )
    rows = inputs["donor_rows"]
    donor_docs = [inputs["documents"][int(row)] for row in rows]
    bag, position, design = _build_design(vectorizer, donor_docs, inputs["donor_r06_oof"])
    residuals = inputs["donor_residuals"]
    if (design.shape[0] != EXPECTED_DONOR_ROWS
            or not np.isfinite(design.data).all() or not np.isfinite(residuals).all()):
        raise BenchmarkError("REAL E79 training features or residual targets are invalid")

    head_params = {**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"]}
    resolved = _resolved_head_parameters()
    if any(resolved.get(key) != value for key, value in head_params.items()):
        raise BenchmarkError("Resolved LGBMRegressor parameters differ from authorization")
    design_sha = _sha_csr(design)
    target_sha = _sha_array(residuals, "<f8")
    head_inputs = {
        "design_csr_sha256": design_sha,
        "target_sha256": target_sha,
        "parameters_sha256": _canonical_sha(resolved),
        "donor_rows_sha256": _sha_array(rows, "<i8"),
        "donor_ids_sha256": _sha_array(inputs["donor_ids"]),
        "donor_final_r06_oof_sha256": _sha_array(inputs["donor_r06_oof"], "<f8"),
    }
    head_meta = {"fit_rows": len(rows), "parameters": resolved}

    def fit_head() -> LGBMRegressor:
        model = LGBMRegressor(**head_params)
        model.fit(design, residuals)
        return model

    head, head_receipt, _ = _fit_unit(
        output_root, FIT_UNITS[1], head_inputs, fit_head, _head_audit, head_meta,
    )
    del head
    ledger = _validate_ledger(output_root)
    if ledger["fit_starts"] != 2 or ledger["fit_saved"] != 2 or ledger["fit_ids"] != list(FIT_UNITS):
        raise BenchmarkError("Production ledger must contain exactly two saved fit starts")
    trained = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "status": "FROZEN_BEFORE_OFFICIAL_TEST_FEATURE_ACCESS",
        "source_root": str(source_root),
        "worktree_root": str(worktree_root),
        "output_root": str(output_root),
        "source_commit_before_fit": _git_head(worktree_root),
        "preflight_sha256": _sha_file(output_root / "preflight.json"),
        "source_freeze_signature": inputs["source_freeze"]["freeze_signature"],
        "input_hashes": preflight["input_hashes"],
        "worktree_source_hashes": preflight["worktree_source_hashes"],
        "official_train": preflight["official_train"],
        "donor_bank": preflight["donor_bank"],
        "vectorizer": {
            "fit_id": FIT_UNITS[0],
            "fit_input_sha256": vectorizer_inputs,
            "artifact_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[0])[0]),
            "receipt_sha256": vector_receipt["receipt_sha256"],
            "receipt_file_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[0])[1]),
            "parameters": preflight["pilot_tfidf"]["resolved_parameters"],
            "sklearn_resolved_parameters": vector_receipt["sklearn_resolved_parameters"],
            "output_audit": vector_receipt["output_audit"],
        },
        "head": {
            "fit_id": FIT_UNITS[1],
            "fit_input_sha256": head_inputs,
            "artifact_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[1])[0]),
            "receipt_sha256": head_receipt["receipt_sha256"],
            "receipt_file_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[1])[1]),
            "parameters": resolved,
            "output_audit": head_receipt["output_audit"],
            "target": "threat_rank - final mapped R06 OOF prediction",
            "features": "[full-Train unigram TF-IDF, REAL three-third position TF-IDF, final R06 OOF/100]",
        },
        "training_feature_hashes": {
            "bag_csr_sha256": _sha_csr(bag),
            "real_position_csr_sha256": _sha_csr(position),
            "design_csr_sha256": design_sha,
            "feature_rows": int(design.shape[0]),
            "feature_columns": int(design.shape[1]),
            "target_sha256": target_sha,
        },
        "accounting": {
            **ledger, "learned_units_total": 2, "fit_starts": 2, "fit_saved": 2,
            "retries": 0, "shared_main_ledger_written": False,
        },
        "policy": {
            "pilot_arm": "REAL",
            "delta": "clip(0.25 * head, -5, +5)",
            "prediction": "clip(final R06 Test prediction + delta, 0, 100)",
            "missing_text": "exact R06 Test prediction",
            "rank_mapping": False,
        },
        "scientific_status": "SCIENTIFIC_GATE_FAILED_EXPLORATORY_BENCHMARK",
        "pilot_verdict": "RETIRE_REG_POSITION_L1_V1",
        "historical_pilot_gates_changed": False,
        "test_features_accessed": False,
        "sample_submission_accessed": False,
        "test_labels_read": False,
        "classification_labels_or_probabilities_read": False,
        "kaggle_accessed": False,
        "created_at_utc": _utc(),
    }
    trained["freeze_signature"] = _canonical_sha(trained)
    _atomic_json(trained_path, trained)
    _verify_trained_freeze(source_root, worktree_root, output_root)
    return trained


def _verify_trained_freeze(source_root: Path, worktree_root: Path, output_root: Path) -> dict[str, Any]:
    path = output_root / "trained_freeze.json"
    freeze = _read_json(path)
    if (not _signature_valid(freeze)
            or freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST_FEATURE_ACCESS"
            or freeze.get("test_features_accessed") is not False
            or freeze.get("sample_submission_accessed") is not False
            or freeze.get("accounting", {}).get("fit_starts") != 2):
        raise BenchmarkError("Trained freeze signature, boundary, or fit count failed")
    if _worktree_hashes(worktree_root) != freeze["worktree_source_hashes"]:
        raise BenchmarkError("Frozen worktree implementation changed")
    for relative, expected in freeze["input_hashes"].items():
        if _sha_file(_contained(source_root, relative)) != expected:
            raise BenchmarkError(f"Frozen source input changed: {relative}")
    for fit_id, section, validator in (
        (FIT_UNITS[0], "vectorizer", _vectorizer_audit),
        (FIT_UNITS[1], "head", _head_audit),
    ):
        state_path, receipt_path = _unit_paths(output_root, fit_id)
        if (_sha_file(state_path) != freeze[section]["artifact_sha256"]
                or _sha_file(receipt_path) != freeze[section]["receipt_file_sha256"]):
            raise BenchmarkError(f"Frozen {fit_id} state/receipt hash failed")
        receipt = _read_json(receipt_path)
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (receipt.get("receipt_sha256") != _canonical_sha(body)
                or receipt.get("receipt_sha256") != freeze[section]["receipt_sha256"]
                or receipt.get("input_sha256") != freeze[section]["fit_input_sha256"]):
            raise BenchmarkError(f"Frozen {fit_id} receipt identity failed")
        payload = joblib.load(state_path)
        if _jsonable(validator(payload, receipt)) != receipt.get("output_audit"):
            raise BenchmarkError(f"Frozen {fit_id} state audit failed")
    ledger = _validate_ledger(output_root)
    if ledger["fit_starts"] != 2 or ledger["fit_saved"] != 2:
        raise BenchmarkError("Frozen production ledger is not exactly two fits")
    return freeze


def _load_r06_test_baseline(source_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_rel = Path("reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json")
    prediction_rel = Path("experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz")
    csv_rel = Path("submissions/optimization_v3_expanded_v1/r06_original.csv")
    manifest_path = _contained(source_root, manifest_rel)
    prediction_path = _contained(source_root, prediction_rel)
    csv_path = _contained(source_root, csv_rel)
    manifest = _read_json(manifest_path)
    if (manifest.get("candidate") != "R06_ORIGINAL"
            or manifest.get("status") != "PREPARED_VALIDATED_NOT_SUBMITTED"
            or manifest.get("test_rows") != EXPECTED_TEST_ROWS
            or manifest.get("sample_submission_rows") != EXPECTED_TEST_ROWS
            or manifest.get("official_input_hashes", {}).get("test") != TEST_SHA256
            or manifest.get("official_input_hashes", {}).get("sample") != SAMPLE_SHA256
            or manifest.get("prediction_artifact", {}).get("path_gitignored") != prediction_rel.as_posix()
            or manifest.get("prediction_artifact", {}).get("sha256") != _sha_file(prediction_path)
            or manifest.get("csv", {}).get("path_gitignored") != csv_rel.as_posix()
            or manifest.get("csv", {}).get("rows") != EXPECTED_TEST_ROWS
            or manifest.get("csv", {}).get("columns") != ["record_id", "threat_rank"]
            or manifest.get("csv", {}).get("sha256") != _sha_file(csv_path)):
        raise BenchmarkError("R06_ORIGINAL manifest or prediction artifact hash failed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        if not {"prediction", "cycle_code"}.issubset(archive.files):
            raise BenchmarkError("R06_ORIGINAL lacks final Test predictions/cycles")
        baseline = np.asarray(archive["prediction"], dtype=np.float64).copy()
        cycles = np.asarray(archive["cycle_code"], dtype=np.int64).copy()
    csv = pd.read_csv(csv_path, dtype={"record_id": "string"})
    values = pd.to_numeric(csv.threat_rank, errors="coerce").to_numpy(dtype=np.float64)
    baseline_hash, cycle_hash = _sha_array(baseline, "<f8"), _sha_array(cycles, "<i8")
    if (csv.columns.tolist() != ["record_id", "threat_rank"] or len(csv) != EXPECTED_TEST_ROWS
            or len(baseline) != EXPECTED_TEST_ROWS
            or not np.isfinite(values).all() or not np.isfinite(baseline).all()
            or np.any((baseline < 0.0) | (baseline > 100.0))
            or np.max(np.abs(values - baseline)) > 1e-12
            or manifest.get("prediction_audit", {}).get("prediction_sha256") != baseline_hash):
        raise BenchmarkError("R06_ORIGINAL final Test prediction values/cycles failed")
    ids = csv.record_id.astype(str).to_numpy(dtype=str)
    if len(baseline) != EXPECTED_TEST_ROWS or len(set(ids.tolist())) != EXPECTED_TEST_ROWS:
        raise BenchmarkError("R06_ORIGINAL Test ID count or uniqueness failed")
    return baseline, cycles, ids, {
        "manifest_path": manifest_rel.as_posix(), "manifest_sha256": _sha_file(manifest_path),
        "prediction_path": prediction_rel.as_posix(), "prediction_file_sha256": _sha_file(prediction_path),
        "prediction_values_sha256": baseline_hash, "csv_path": csv_rel.as_posix(),
        "csv_sha256": _sha_file(csv_path), "cycle_values_sha256": cycle_hash,
    }


def _read_test_features(
    source_root: Path, expected_ids: np.ndarray, expected_cycles: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, str]]:
    test_path = _contained(source_root, TEST_REL)
    sample_path = _contained(source_root, SAMPLE_REL)
    if _sha_file(test_path) != TEST_SHA256 or _sha_file(sample_path) != SAMPLE_SHA256:
        raise BenchmarkError("Official Test/Sample hashes differ from frozen source")
    test = pd.read_csv(test_path, usecols=list(FEATURE_COLUMNS), dtype={"record_id": "string"})
    test = test.loc[:, list(FEATURE_COLUMNS)].copy()
    sample = pd.read_csv(sample_path, usecols=["record_id"], dtype={"record_id": "string"})
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    cycles_values = pd.to_numeric(test.cycle_code, errors="coerce").to_numpy()
    sample_ids = sample.record_id.astype(str).to_numpy(dtype=str)
    if (len(test) != EXPECTED_TEST_ROWS or len(sample_ids) != EXPECTED_TEST_ROWS
            or len(set(ids.tolist())) != EXPECTED_TEST_ROWS
            or len(set(sample_ids.tolist())) != EXPECTED_TEST_ROWS
            or not np.isfinite(cycles_values).all() or np.any(cycles_values != np.floor(cycles_values))
            or not np.array_equal(ids, sample_ids) or not np.array_equal(ids, expected_ids)
            or not np.array_equal(cycles_values.astype(np.int64), expected_cycles)):
        raise BenchmarkError("Official Test, SampleSubmission, and R06 identity/order differ")
    test["cycle_code"] = cycles_values.astype(np.int64)
    return test, {TEST_REL.as_posix(): _sha_file(test_path),
                  SAMPLE_REL.as_posix(): _sha_file(sample_path)}


def _compose_prediction(baseline: np.ndarray, head: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline, dtype=np.float64)
    head = np.asarray(head, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if (baseline.ndim != 1 or head.shape != baseline.shape or missing.shape != baseline.shape
            or not np.isfinite(baseline).all() or not np.isfinite(head).all()
            or np.any((baseline < 0.0) | (baseline > 100.0))):
        raise BenchmarkError("Test baseline/head/missing arrays are invalid")
    delta = np.clip(0.25 * head, -5.0, 5.0)
    delta[missing] = 0.0
    prediction = np.clip(baseline + delta, 0.0, 100.0)
    prediction[missing] = baseline[missing]
    if (not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 100.0))
            or not np.array_equal(prediction[missing], baseline[missing])):
        raise BenchmarkError("Fixed REAL residual policy failed")
    return prediction, delta


def _write_csv_once(path: Path, ids: np.ndarray, prediction: np.ndarray) -> str:
    frame = pd.DataFrame({"record_id": np.asarray(ids, dtype=str),
                          "threat_rank": np.asarray(prediction, dtype=np.float64)})
    payload = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    _atomic_bytes(path, payload)
    return _sha_file(path)


def _scan_duplicate_vectors(
    worktree_root: Path, source_root: Path, ids: np.ndarray,
    values: np.ndarray, candidate: Path,
) -> dict[str, Any]:
    value_sha = _sha_array(values, "<f8")
    pair_sha = _canonical_sha(sorted(zip(np.asarray(ids, dtype=str).tolist(),
                                         np.asarray(values, dtype=np.float64).tolist())))
    scanned, compatible = 0, 0
    same_vector, same_pairs = [], []
    for base in (worktree_root / "submissions", source_root / "submissions"):
        if not base.is_dir():
            continue
        for path in base.rglob("*.csv"):
            if path.resolve() == candidate.resolve():
                continue
            scanned += 1
            try:
                frame = pd.read_csv(path, dtype={"record_id": "string"})
            except Exception:
                continue
            if not {"record_id", "threat_rank"}.issubset(frame.columns):
                continue
            labels = pd.to_numeric(frame.threat_rank, errors="coerce").to_numpy(dtype=np.float64)
            other_ids = frame.record_id.astype(str).to_numpy(dtype=str)
            if (len(labels) != len(values) or len(other_ids) != len(ids)
                    or not np.isfinite(labels).all()):
                continue
            compatible += 1
            if _sha_array(labels, "<f8") == value_sha:
                same_vector.append(str(path))
            if _canonical_sha(sorted(zip(other_ids.tolist(), labels.tolist()))) == pair_sha:
                same_pairs.append(str(path))
    return {
        "files_scanned": scanned,
        "schema_compatible_files": compatible,
        "candidate_value_vector_sha256": value_sha,
        "candidate_record_id_value_pairs_sha256": pair_sha,
        "same_ordered_value_vectors": sorted(same_vector),
        "same_record_id_value_pairs": sorted(same_pairs),
    }


def run_inference(
    source_root: Path, worktree_root: Path, output_root: Path, csv_path: Path | None = None,
) -> dict[str, Any]:
    from src.models.classification_models import _text_values

    source_root, worktree_root = source_root.resolve(), worktree_root.resolve()
    _check_roots(source_root, worktree_root)
    output_root = _run_root(worktree_root, output_root)
    freeze = _verify_trained_freeze(source_root, worktree_root, output_root)
    # The R06 manifest and saved baseline are opened before official Test features.
    baseline, baseline_cycles, baseline_ids, baseline_audit = _load_r06_test_baseline(source_root)
    test, test_hashes = _read_test_features(source_root, baseline_ids, baseline_cycles)
    documents, missing = _text_values(test.signal_stream)
    vectorizer = joblib.load(_unit_paths(output_root, FIT_UNITS[0])[0])
    head_model = joblib.load(_unit_paths(output_root, FIT_UNITS[1])[0])
    bag, position, design = _build_design(vectorizer, documents, baseline)
    with threadpool_limits(limits=THREADS):
        head = np.asarray(head_model.predict(design), dtype=np.float64)
    prediction, delta = _compose_prediction(baseline, head, missing)
    if len(prediction) != EXPECTED_TEST_ROWS:
        raise BenchmarkError("Test prediction row count differs")
    inference_dir = output_root / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = inference_dir / "predictions.npz"
    arrays = {
        "record_ids": test.record_id.astype(str).to_numpy(dtype=str),
        "cycle_code": test.cycle_code.to_numpy(dtype=np.int64),
        "baseline": baseline, "missing": np.asarray(missing, dtype=bool),
        "head_prediction": head, "delta": delta, "prediction": prediction,
    }
    _save_npz_once(bundle_path, arrays)
    if csv_path is None:
        csv_path = worktree_root / "submissions/optimization_v3_reg_position_real_e79_exploratory_benchmark_v1.csv"
    csv_path = csv_path.resolve()
    if worktree_root not in csv_path.parents:
        raise BenchmarkError("Submission CSV must be inside the isolated worktree")
    csv_sha = _write_csv_once(csv_path, arrays["record_ids"], prediction)
    manifest = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "status": "PREPARED_NOT_UPLOADED",
        "scientific_status": "SCIENTIFIC_GATE_FAILED_EXPLORATORY_BENCHMARK",
        "pilot_verdict": "RETIRE_REG_POSITION_L1_V1",
        "historical_pilot_gates_changed": False,
        "selected_arm": "REAL",
        "selection_basis": "fixed by authorization before this benchmark; no diagnostic-derived alternative",
        "known_parent_r06_public_score": {
            "value": 11.16620,
            "status": "root-provided historical reference only",
            "this_benchmark_score": False,
        },
        "upload_label_for_root": FINAL_UPLOAD_LABEL,
        "upload_owner": "root Sol only",
        "upload_performed": False,
        "trained_freeze_sha256": _sha_file(_trained_freeze_path(output_root)),
        "trained_freeze_signature": freeze["freeze_signature"],
        "baseline": baseline_audit,
        "test_input_hashes": test_hashes,
        "test_rows": len(test),
        "test_ids_sha256": _sha_array(arrays["record_ids"]),
        "test_cycles_sha256": _sha_array(arrays["cycle_code"], "<i8"),
        "feature_hashes": {
            "bag_csr_sha256": _sha_csr(bag),
            "real_position_csr_sha256": _sha_csr(position),
            "design_csr_sha256": _sha_csr(design),
            "design_rows": int(design.shape[0]),
            "design_columns": int(design.shape[1]),
        },
        "prediction_artifact": {
            "path": bundle_path.relative_to(worktree_root).as_posix(),
            "sha256": _sha_file(bundle_path),
            "prediction_sha256": _sha_array(prediction, "<f8"),
            "head_sha256": _sha_array(head, "<f8"),
            "delta_sha256": _sha_array(delta, "<f8"),
            "missing_rows": int(np.sum(missing)),
            "exact_r06_missing_fallback": bool(np.array_equal(prediction[missing], baseline[missing])),
            "range": [float(prediction.min()), float(prediction.max())],
        },
        "submission": {
            "path": csv_path.relative_to(worktree_root).as_posix(),
            "sha256": csv_sha, "rows": len(test),
            "columns": ["record_id", "threat_rank"], "float_format": "%.17g",
        },
        "duplicate_label_vector_scan": _scan_duplicate_vectors(
            worktree_root, source_root, arrays["record_ids"], prediction, csv_path,
        ),
        "boundaries": {
            "test_labels_read": False, "classification_data_read": False,
            "kaggle_accessed": False, "main_source_tree_written": False,
            "shared_main_ledger_written": False,
        },
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    _atomic_json(inference_dir / "manifest.json", manifest)
    _atomic_json(worktree_root / REPORT_REL / "INFERENCE_MANIFEST.json", manifest)
    return manifest


def _trained_freeze_path(output_root: Path) -> Path:
    return output_root / "trained_freeze.json"


def _synthetic_e2e() -> dict[str, Any]:
    from src.models.classification_models import _TFIDF_PARAMS
    from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS

    with tempfile.TemporaryDirectory(prefix="reg-position-e79-synthetic-") as tmp:
        temp_root = Path(tmp).resolve()
        output_root = temp_root / "isolated" / RUN_REL
        output_root.mkdir(parents=True)
        count = 288
        cycles = np.asarray([47 + (i % 33) for i in range(count)], dtype=np.int64)
        documents = [
            f"alpha beta token{i % 9} group{i % 5} cycle{cycles[i]} message{i % 11} delta{i % 7}"
            for i in range(count)
        ]
        donor_rows = np.flatnonzero(cycles >= 56)
        donor_documents = [documents[int(i)] for i in donor_rows]
        donor_baseline = 20.0 + (donor_rows % 60).astype(np.float64)
        donor_target = ((donor_rows % 17).astype(np.float64) - 8.0) / 2.0
        fixture = {
            "train_documents": documents,
            "donor_documents": donor_documents,
            "donor_rows": donor_rows.tolist(),
            "donor_baseline": donor_baseline.tolist(),
            "donor_target": donor_target.tolist(),
            "test_documents": [
                "alpha beta token2 group1 message7", "gamma epsilon token5 group2",
                "delta alpha token0 group4", "beta beta token8 message1",
            ],
            "test_ids": ["synthetic-test-a", "synthetic-test-b", "synthetic-test-c", "synthetic-test-d"],
            "test_baseline": [22.0, 45.0, 74.0, 91.0],
            "test_missing": [False, False, True, False],
        }
        fixture_path = temp_root / "fixture.json"
        _atomic_json(fixture_path, fixture)
        _atomic_json(output_root / "preflight.json", {"synthetic_only": True})
        _append_event(output_root, {
            "at_utc": _utc(), "event": "FIT_BUDGET_RESERVED",
            "fit_ids": list(FIT_UNITS), "card_id": CARD_ID, "synthetic_only": True,
        })
        calls: list[str] = []
        vectorizer_inputs = {"synthetic_documents_sha256": _canonical_sha(documents)}
        vectorizer_meta = {
            "fit_rows": len(documents),
            "parameters": _normalize_tfidf_parameters(_TFIDF_PARAMS),
            "sklearn_resolved_parameters": _resolved_tfidf_parameters(_TFIDF_PARAMS),
        }
        vectorizer, _, _ = _fit_unit(
            output_root, FIT_UNITS[0], vectorizer_inputs,
            lambda: _fit_vectorizer(documents, _TFIDF_PARAMS),
            _vectorizer_audit, vectorizer_meta, synthetic=True, call_log=calls,
        )
        bag, position, design = _build_design(vectorizer, donor_documents, donor_baseline)
        params = _resolved_head_parameters()
        head_inputs = {
            "design_csr_sha256": _sha_csr(design),
            "target_sha256": _sha_array(donor_target, "<f8"),
            "parameters_sha256": _canonical_sha(params),
        }
        head_meta = {"fit_rows": len(donor_rows), "parameters": params}

        def fit_synthetic_head() -> LGBMRegressor:
            model = LGBMRegressor(**{**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"]})
            model.fit(design, donor_target)
            return model

        head_model, _, _ = _fit_unit(
            output_root, FIT_UNITS[1], head_inputs, fit_synthetic_head,
            _head_audit, head_meta, synthetic=True, call_log=calls,
        )
        before_resume = _sha_file(output_root / "fit_ledger.jsonl")
        _, _, resumed_vec = _fit_unit(
            output_root, FIT_UNITS[0], vectorizer_inputs,
            lambda: _fit_vectorizer(documents, _TFIDF_PARAMS),
            _vectorizer_audit, vectorizer_meta, synthetic=True, call_log=calls,
        )
        _, _, resumed_head = _fit_unit(
            output_root, FIT_UNITS[1], head_inputs, fit_synthetic_head,
            _head_audit, head_meta, synthetic=True, call_log=calls,
        )
        if (not resumed_vec or not resumed_head or calls != list(FIT_UNITS)
                or before_resume != _sha_file(output_root / "fit_ledger.jsonl")):
            raise BenchmarkError("Synthetic persisted resume refit or changed its ledger")
        test_base = np.asarray(fixture["test_baseline"], dtype=np.float64)
        test_missing = np.asarray(fixture["test_missing"], dtype=bool)
        _, _, test_design = _build_design(vectorizer, fixture["test_documents"], test_base)
        with threadpool_limits(limits=THREADS):
            test_head = np.asarray(head_model.predict(test_design), dtype=np.float64)
        prediction, delta = _compose_prediction(test_base, test_head, test_missing)
        infer_dir = output_root / "synthetic_inference"
        infer_dir.mkdir()
        arrays = {
            "record_ids": np.asarray(fixture["test_ids"], dtype=str),
            "baseline": test_base, "missing": test_missing,
            "head_prediction": test_head, "delta": delta, "prediction": prediction,
        }
        bundle = infer_dir / "predictions.npz"
        _save_npz_once(bundle, arrays)
        frame = pd.DataFrame({"record_id": fixture["test_ids"], "threat_rank": prediction})
        csv_path = infer_dir / "submission.csv"
        csv_path.write_text(
            frame.to_csv(index=False, float_format="%.17g", lineterminator="\n"),
            encoding="utf-8", newline="",
        )
        manifest = {
            "vectorizer_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[0])[0]),
            "head_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[1])[0]),
            "vectorizer_receipt_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[0])[1]),
            "head_receipt_sha256": _sha_file(_unit_paths(output_root, FIT_UNITS[1])[1]),
            "bundle_sha256": _sha_file(bundle), "csv_sha256": _sha_file(csv_path),
            "test_design_sha256": _sha_csr(test_design),
        }
        _atomic_json(infer_dir / "manifest.json", manifest)
        verifier = ROOT / "scripts/verify_reg_position_real_e79_benchmark_v1.py"
        result = subprocess.run(
            [sys.executable, str(verifier), "--synthetic-root", str(temp_root)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise BenchmarkError(f"Synthetic independent verifier failed: {result.stdout} {result.stderr}")
        verified = json.loads(result.stdout.strip().splitlines()[-1])
        if verified.get("status") != "PASS_SYNTHETIC_INDEPENDENT_VERIFIER":
            raise BenchmarkError("Synthetic independent verifier did not pass")
        ledger = _validate_ledger(output_root)
        if ledger["fit_starts"] != 2 or ledger["fit_saved"] != 2:
            raise BenchmarkError("Synthetic fit ledger does not contain exactly two fit units")
        return {
            "status": "PASS_SYNTHETIC_E2E", "temp_only": True,
            "fit_invocations": calls, "fit_starts": ledger["fit_starts"],
            "fit_saved": ledger["fit_saved"], "resume_without_refit": True,
            "independent_verifier": verified["status"],
            "inference_rows": len(fixture["test_ids"]),
            "temporary_root_removed_on_exit": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "fit", "infer", "synthetic-e2e"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--worktree-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--csv-path", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "synthetic-e2e":
            result = _synthetic_e2e()
        else:
            if args.source_root is None or args.output_root is None:
                raise BenchmarkError("--source-root and --output-root are required")
            if args.command == "preflight":
                result = run_preflight(args.source_root, args.worktree_root, args.output_root)
            elif args.command == "fit":
                result = run_fit(args.source_root, args.worktree_root, args.output_root)
            else:
                result = run_inference(args.source_root, args.worktree_root, args.output_root, args.csv_path)
        print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL_CLOSED", "error_type": type(exc).__name__, "error": str(exc),
        }, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
