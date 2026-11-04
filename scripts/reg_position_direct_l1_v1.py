"""Frozen two-origin direct position regression pilot for KASDAD Task 2.

The official fit path is deliberately split into preflight, fit, and score.
Preflight reads features and frozen prediction/cache metadata only. The fit
path reads targets only for each origin's own training prefix. Query targets
remain closed until six models and their raw predictions pass prelabel QA.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from scipy import sparse
from scipy.stats import rankdata
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
TRAIN_REL = Path("data/raw/regression/Train.csv")
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
CLASSIFICATION_TRAIN_REL = Path("data/raw/classification/Train.csv")
CARD_ID = "REG_POSITION_DIRECT_L1_V1"
ORIGINS = (67, 75)
ARMS = ("BAG", "REAL", "SHUFFLED")
EXPECTED = {
    67: {"fit_rows": 29_705, "query_rows": 20_903, "fit_cycles": (47, 67), "query_cycles": (68, 79)},
    75: {"fit_rows": 42_117, "query_rows": 8_491, "fit_cycles": (47, 75), "query_cycles": (76, 79)},
}
RUN_REL = Path("experiments/runs/reg_position_direct_l1_v1")
REPORT_REL = Path("reports/optimization_v3/reg_position_direct_l1_v1")
RECIPE_REL = REPORT_REL / "RECIPE.json"
RUN_ROOT = ROOT / RUN_REL
REPORT_ROOT = ROOT / REPORT_REL
FREEZE_PATH = RUN_ROOT / "FREEZE_V4.json"
SUPERSEDED_FREEZE_PATH = RUN_ROOT / "FREEZE_V3.json"
LEDGER_PATH = RUN_ROOT / "fit_ledger.jsonl"
PREFIT_RUN_PATH = RUN_ROOT / "PREFIT_VERIFICATION_V4.json"
PREFIT_REPORT_PATH = REPORT_ROOT / "PREFIT_VERIFICATION_V4.json"
PRELABEL_PATH = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
POSITION_RUN_REL = Path("experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1")
POSITION_BUDGET_REL = Path("experiments/runs/optimization_v3_research_next_v1/budget.jsonl")
TEXT_RUN_REL = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V2_1/text_space_replays")
OOF_CACHE_REL = {
    67: Path("experiments/runs/optimization_v3_batch1/R06/R06_fit67.npz"),
    75: Path("experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit75.npz"),
}
POSITION_FREEZE_SIGNATURE = "f310d0340ea4402677ca9e86aa29c43b1536ded902c0f6a6f52cb53bd333c292"
POSITION_INPUT_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
MIN_AVAILABLE_RAM = 4 * 1024**3
MIN_FREE_RAM_RESERVE = 2 * 1024**3
MAX_RSS = 6 * 1024**3
THREADS = 2
TOL = 1e-10

from src.models.classification_models import _text_values  # noqa: E402
from src.models.optimization_v3_positional_text_pilot_v1 import (  # noqa: E402
    build_position_block,
    canonical_sha256,
    sha256_array as _position_sha_array,
    sha256_csr as _position_sha_csr,
)
from src.models.reg_position_l1_v1 import unigram_bag  # noqa: E402
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS  # noqa: E402
from src.models.regression_research_text import (  # noqa: E402
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
)
from src.models.round1_models import CATEGORICAL_FEATURES, _is_missing  # noqa: E402
from src.models.regression_round2_models import _SelectedPreprocessor  # noqa: E402


class PilotError(RuntimeError):
    """An input, accounting, or integrity gate failed closed."""


FEATURE_COLUMNS = tuple(dict.fromkeys(("record_id", "cycle_code", *HYBRID_FEATURES)))
FIT_IDS = tuple(
    f"{CARD_ID}_E{origin}_{component}"
    for origin in ORIGINS
    for component in ("STRUCTURED", "SCALER", "OHE", *ARMS)
)
SOURCE_FILES = (
    "scripts/reg_position_direct_l1_v1.py",
    "scripts/verify_reg_position_direct_l1_v1.py",
    "tests/test_reg_position_direct_l1_v1.py",
    "reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/SYNTHETIC_E2E.json",
    "src/models/reg_position_l1_v1.py",
    "src/models/regression_research_text.py",
    "src/models/regression_research_tabular.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/classification_models.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/positional_text_features.py",
)


@dataclass
class OriginInputs:
    origin: int
    fit_rows: np.ndarray
    query_rows: np.ndarray
    fit_ids: np.ndarray
    query_ids: np.ndarray
    fit_cycles: np.ndarray
    query_cycles: np.ndarray
    fit_missing: np.ndarray
    query_missing: np.ndarray
    fit_bag: sparse.csr_matrix
    query_bag: sparse.csr_matrix
    fit_position: sparse.csr_matrix
    query_position: sparse.csr_matrix
    fit_shuffled: sparse.csr_matrix
    query_shuffled: sparse.csr_matrix
    fit_baseline: np.ndarray
    query_baseline: np.ndarray
    group_features: dict[str, np.ndarray]
    group_edges: dict[str, list[float]]
    audit: dict[str, Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_raw_array(value: Any, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    header = _canonical({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header)
    if array.dtype.kind in "OUS":
        digest.update(_canonical(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _sha_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(_canonical_sha({"shape": matrix.shape, "dtype": str(matrix.dtype)}).encode("ascii"))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def _sha_standalone_vocabulary(vectorizer: Any) -> str:
    vocab = sorted(((str(term), int(index)) for term, index in vectorizer.vocabulary_.items()),
                   key=lambda item: item[1])
    return _canonical_sha(vocab)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary, compress=0)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contained_file(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise PilotError(f"Input path escapes its declared root: {relative}")
    if not path.exists():
        raise PilotError(f"Required input is missing: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"Expected a JSON object: {path}")
    return value


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def _resolved_parameters() -> dict[str, Any]:
    from lightgbm import LGBMRegressor
    params = {**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"]}
    resolved = LGBMRegressor(**params).get_params(deep=False)
    return {key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in sorted(resolved.items())}


def _load_feature_frame(source_root: Path) -> pd.DataFrame:
    train = _contained_file(source_root, TRAIN_REL)
    if _sha_file(train) != TRAIN_SHA256:
        raise PilotError("Official Regression Train hash differs from the frozen source")
    frame = pd.read_csv(train, usecols=list(FEATURE_COLUMNS), dtype={"record_id": "string"})
    frame = frame.loc[:, list(FEATURE_COLUMNS)]
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    cycle_values = pd.to_numeric(frame.cycle_code, errors="coerce").to_numpy(dtype=np.float64)
    if (len(frame) != 50_608 or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(cycle_values).all()
            or not np.equal(cycle_values, np.floor(cycle_values)).all()):
        raise PilotError("Regression feature row count, IDs, or cycles failed integrity checks")
    frame["cycle_code"] = cycle_values.astype(np.int64)
    if (int(frame.cycle_code.min()) != 47 or int(frame.cycle_code.max()) != 79
            or frame.columns.has_duplicates or "threat_rank" in frame.columns):
        raise PilotError("Regression feature schema or cycle bounds differ from the frozen scope")
    friction = pd.to_numeric(frame.entry_friction, errors="coerce")
    if not friction.dropna().isin([0, 1]).all():
        raise PilotError("entry_friction contains values outside the frozen {0,1} schema")
    return frame


def _verify_classification_regression_alignment(source_root: Path,
                                               regression_frame: pd.DataFrame) -> dict[str, Any]:
    """Confirm source position caches' Classification text rows equal Regression rows."""
    classification_path = _contained_file(source_root, CLASSIFICATION_TRAIN_REL)
    if _sha_file(classification_path) != POSITION_INPUT_SHA256:
        raise PilotError("Classification Train hash differs from the positional cache source freeze")
    columns = ["record_id", "cycle_code", "signal_stream"]
    classification = pd.read_csv(classification_path, usecols=columns,
                                 dtype={"record_id": "string"})
    classification["cycle_code"] = pd.to_numeric(classification.cycle_code, errors="raise").astype(np.int64)
    regression_ids = regression_frame.record_id.astype(str).to_numpy(dtype=str)
    classification_ids = classification.record_id.astype(str).to_numpy(dtype=str)
    regression_cycles = regression_frame.cycle_code.to_numpy(dtype=np.int64)
    classification_cycles = classification.cycle_code.to_numpy(dtype=np.int64)
    regression_docs, regression_missing = _text_values(regression_frame.signal_stream)
    classification_docs, classification_missing = _text_values(classification.signal_stream)
    if (len(classification) != len(regression_frame)
            or not np.array_equal(classification_ids, regression_ids)
            or not np.array_equal(classification_cycles, regression_cycles)
            or classification_docs != regression_docs
            or not np.array_equal(classification_missing, regression_missing)):
        raise PilotError("Classification and Regression ordered IDs/cycles/text/missing masks differ")
    return {
        "classification_train_sha256": _sha_file(classification_path),
        "regression_train_sha256": _sha_file(_contained_file(source_root, TRAIN_REL)),
        "rows": len(classification),
        "ordered_ids_equal": True,
        "cycles_equal": True,
        "normalized_text_documents_equal": True,
        "missing_masks_equal": True,
        "missing_rows": int(regression_missing.sum()),
        "record_ids_sha256": _sha_array(regression_ids),
        "cycles_sha256": _sha_array(regression_cycles),
        "normalized_documents_sha256": _canonical_sha(regression_docs),
        "target_columns_loaded": False,
    }


def _prefix_rows(frame: pd.DataFrame, origin: int) -> tuple[np.ndarray, np.ndarray]:
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero(cycles <= origin).astype(np.int64)
    query_rows = np.flatnonzero(cycles > origin).astype(np.int64)
    expected = EXPECTED[origin]
    if (len(fit_rows) != expected["fit_rows"] or len(query_rows) != expected["query_rows"]
            or tuple((int(cycles[fit_rows].min()), int(cycles[fit_rows].max()))) != expected["fit_cycles"]
            or tuple((int(cycles[query_rows].min()), int(cycles[query_rows].max()))) != expected["query_cycles"]
            or not np.array_equal(np.sort(np.concatenate((fit_rows, query_rows))),
                                  np.arange(len(frame), dtype=np.int64))):
        raise PilotError(f"E{origin} source rows do not match the frozen origin boundary")
    return fit_rows, query_rows


def _verify_text_state(source_root: Path, origin: int, frame: pd.DataFrame,
                       fit_rows: np.ndarray) -> tuple[Any, dict[str, Any]]:
    text_dir = _contained_file(source_root, TEXT_RUN_REL / f"E{origin}")
    vector_path = _contained_file(text_dir, "vectorizer.joblib")
    state_path = _contained_file(text_dir, "text_state.json")
    receipt_path = _contained_file(text_dir, "receipt.json")
    receipt = _read_json(receipt_path)
    state = _read_json(state_path)
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
    docs, _ = _text_values(frame.iloc[fit_rows].signal_stream)
    docs_sha = _canonical_sha(docs)
    rows_sha = _sha_raw_array(fit_rows, dtype="<i8")
    ids = frame.iloc[fit_rows].record_id.astype(str).to_numpy()
    cycle = frame.iloc[fit_rows].cycle_code.to_numpy(dtype=np.int64)
    population = receipt.get("fit_population", {})
    if (state.get("origin_cycle") != origin or state_audit != audit
            or population.get("rows") != len(fit_rows)
            or population.get("cycle_bounds") != [47, origin]
            or population.get("row_positions_sha256") != rows_sha
            or population.get("documents_sha256") != docs_sha
            or audit.get("fit_rows") != len(fit_rows)
            or audit.get("fit_row_positions_sha256") != rows_sha
            or audit.get("fit_documents_sha256") != docs_sha
            or tuple((int(cycle.min()), int(cycle.max()))) != (47, origin)):
        raise PilotError(f"E{origin} target-free vectorizer row/document population differs")
    vectorizer = joblib.load(vector_path)
    if not getattr(vectorizer, "vocabulary_", None) or not hasattr(vectorizer, "idf_"):
        raise PilotError(f"E{origin} standalone word vectorizer state is incomplete")
    if (_sha_standalone_vocabulary(vectorizer) != audit.get("vocabulary_sha256")
            or _sha_raw_array(vectorizer.idf_, dtype="<f8") != audit.get("idf_sha256")
            or len(vectorizer.vocabulary_) != audit.get("tfidf_vocabulary_size")):
        raise PilotError(f"E{origin} standalone vocabulary/IDF receipt failed")
    params = vectorizer.get_params(deep=False)
    params = dict(params)
    params["dtype"] = np.dtype(params["dtype"]).name
    expected_params = audit.get("tfidf_parameters")
    for key in ("ngram_range", "min_df", "max_df", "max_features", "sublinear_tf",
                "lowercase", "strip_accents", "dtype"):
        expected = expected_params.get(key) if isinstance(expected_params, dict) else None
        actual = list(params[key]) if key == "ngram_range" else params[key]
        if actual != expected:
            raise PilotError(f"E{origin} standalone TF-IDF parameter differs: {key}")
    return vectorizer, {
        "vectorizer_sha256": receipt["vectorizer_sha256"],
        "text_state_sha256": receipt["text_state_sha256"],
        "receipt_sha256": _sha_file(receipt_path),
        "fit_rows": len(fit_rows), "fit_ids_sha256": _sha_array(ids),
        "fit_cycles_sha256": _sha_array(cycle),
        "fit_row_positions_raw_sha256": rows_sha,
        "documents_sha256": docs_sha,
        "vocabulary_sha256": audit["vocabulary_sha256"],
        "idf_raw_f64_sha256": audit["idf_sha256"],
        "fit_transform_calls": receipt["fit_transform_calls"],
        "query_targets_read": False,
    }


def _verify_positional_vectorizer(source_root: Path, origin: int, standalone: Any,
                                  fit_rows: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    base = _contained_file(source_root, POSITION_RUN_REL)
    receipt_path = _contained_file(base, f"receipts/POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF.json")
    receipt = _read_json(receipt_path)
    artifact_path = _contained_file(source_root, receipt["artifact_path"])
    if (_sha_file(artifact_path) != receipt.get("artifact_sha256")
            or receipt.get("schema_version") != 1
            or receipt.get("fit_id") != f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF"
            or receipt.get("component") != "C2B_TFIDF"
            or receipt.get("origin") != origin
            or receipt.get("freeze_signature") != POSITION_FREEZE_SIGNATURE):
        raise PilotError(f"E{origin} original positional vectorizer artifact hash mismatch")
    budget_path = _contained_file(source_root, POSITION_BUDGET_REL)
    fit_id = receipt["fit_id"]
    saved_events = []
    started_events = []
    with budget_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if fit_id not in line:
                continue
            try:
                event = json.loads(line)
            except Exception as exc:
                raise PilotError(f"Invalid shared fit accounting line {line_no}") from exc
            if event.get("fit_id") != fit_id:
                continue
            if event.get("event") == "FIT_STARTED":
                started_events.append(event)
            elif event.get("event") == "FIT_SAVED":
                saved_events.append(event)
    if (len(started_events) != 1 or len(saved_events) != 1
            or saved_events[0].get("receipt_sha256") != _sha_file(receipt_path)
            or started_events[0].get("freeze_signature") != POSITION_FREEZE_SIGNATURE
            or receipt.get("freeze_signature") != POSITION_FREEZE_SIGNATURE
            or started_events[0].get("fit_population") != receipt.get("fit_population")):
        raise PilotError(f"E{origin} C2B_TFIDF receipt does not reconcile to FIT_ACCOUNTING")
    original = joblib.load(artifact_path)
    meta = receipt.get("fit_metadata", {})
    population = receipt.get("fit_population", {})
    documents, _ = _text_values(frame.iloc[fit_rows].signal_stream)
    ids = frame.iloc[fit_rows].record_id.astype(str).to_numpy()
    cycles = frame.iloc[fit_rows].cycle_code.to_numpy(dtype=np.int64)
    vocab_hash = _sha_standalone_vocabulary(standalone)
    same = (
        np.array_equal(np.asarray(standalone.idf_), np.asarray(original.idf_))
        and dict(standalone.vocabulary_) == dict(original.vocabulary_)
        and standalone.get_params(deep=False) == original.get_params(deep=False)
    )
    if not same:
        raise PilotError(f"E{origin} standalone and positional TF-IDF numerical states differ")
    if (meta.get("idf_sha256") != _position_sha_array(original.idf_)
            or meta.get("vocabulary_sha256") != vocab_hash
            or meta.get("feature_count") != len(original.vocabulary_)
            or population.get("rows") != len(fit_rows)
            or population.get("row_indices_sha256") != _position_sha_array(fit_rows)
            or population.get("record_ids_sha256") != _position_sha_array(ids)
            or population.get("cycles_sha256") != _position_sha_array(cycles)
            or population.get("cycle_min") != 47 or population.get("cycle_max") != origin):
        raise PilotError(f"E{origin} positional TF-IDF header-aware digest/population check failed")
    return {
        "artifact_sha256": receipt["artifact_sha256"],
        "receipt_sha256": _sha_file(receipt_path),
        "fit_accounting_sha256": _sha_file(budget_path),
        "fit_accounting_saved_receipt_sha256": saved_events[0]["receipt_sha256"],
        "vocabulary_sha256": vocab_hash,
        "idf_header_aware_f64_sha256": _position_sha_array(original.idf_),
        "fit_population": population,
        "numeric_state_equal_to_standalone": True,
        "max_abs_idf_difference": 0.0,
        "vocabulary_dict_equal": True,
        "parameters_equal": True,
        "fit_documents_sha256": _canonical_sha(documents),
    }


def _manifest_path(source_root: Path, manifest: Mapping[str, Any], key: str) -> Path:
    item = manifest.get("matrices", {}).get(key)
    if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
        raise PilotError(f"Positional matrix manifest lacks {key}")
    path = _contained_file(source_root, item["path"])
    if _sha_file(path) != item["sha256"]:
        raise PilotError(f"Positional matrix file hash mismatch: {key}")
    return path


def _load_matrix_set(source_root: Path, manifest: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, sparse.csr_matrix]:
    output: dict[str, sparse.csr_matrix] = {}
    for key in keys:
        item = manifest["matrices"][key]
        path = _manifest_path(source_root, manifest, key)
        if path.suffix != ".npz":
            raise PilotError(f"Expected sparse NPZ matrix for {key}")
        matrix = sparse.load_npz(path).tocsr()
        if (list(matrix.shape) != item.get("shape")
                or _sha_csr(matrix) != item.get("logical_sha256")
                or not np.isfinite(matrix.data).all()):
            raise PilotError(f"Positional matrix shape/logical hash/values failed: {key}")
        output[key] = matrix
    if _canonical_sha(manifest["matrices"]) != manifest.get("matrix_set_sha256"):
        raise PilotError("Positional matrix set aggregate hash failed")
    return output


def _same_csr(left: sparse.spmatrix, right: sparse.spmatrix, *, atol: float = 0.0) -> bool:
    a = sparse.csr_matrix(left, dtype=np.float64, copy=True)
    b = sparse.csr_matrix(right, dtype=np.float64, copy=True)
    a.sum_duplicates(); a.sort_indices()
    b.sum_duplicates(); b.sort_indices()
    if a.shape != b.shape:
        return False
    diff = a - b
    return bool(diff.nnz == 0 or np.max(np.abs(diff.data)) <= atol)


def _sample_position_reconstruction(frame: pd.DataFrame, rows: np.ndarray, cached: sparse.csr_matrix,
                                    vectorizer: Any, *, shuffled: bool, seed: int) -> dict[str, Any]:
    docs, missing = _text_values(frame.iloc[rows].signal_stream)
    absent = np.flatnonzero(missing)
    present = np.flatnonzero(~missing)
    if len(present) < 63:
        raise PilotError("Position reconstruction needs at least 63 present text rows")
    rng = np.random.default_rng(seed)
    selected = np.sort(np.concatenate((absent[:1], rng.choice(present, size=63, replace=False)))).astype(np.int64)
    rebuilt, _ = build_position_block([docs[i] for i in selected], vectorizer,
                                     shuffled=shuffled, seed=42)
    reference = sparse.csr_matrix(cached[selected], dtype=np.float64)
    if not _same_csr(rebuilt, reference, atol=1e-12):
        raise PilotError("Cached three-third position suffix differs from target-free reconstruction")
    return {"sample_rows": len(selected), "missing_row_included": bool(len(absent)),
            "arm": "SHUFFLED" if shuffled else "REAL", "equal_atol_1e-12": True,
            "cached_sample_sha256": _sha_csr(reference), "reconstructed_sha256": _sha_csr(rebuilt)}


def _read_oof_cache(source_root: Path, frame: pd.DataFrame, origin: int,
                    query_rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    path = _contained_file(source_root, OOF_CACHE_REL[origin])
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive["score_rows"], dtype=np.int64)
        ids = np.asarray(archive["score_record_ids"], dtype=str)
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
    all_ids = frame.record_id.astype(str).to_numpy(dtype=str)
    all_cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    if (rows.ndim != 1 or ids.shape != rows.shape
            or prediction.shape != rows.shape or len(set(rows.tolist())) != len(rows)
            or np.any((rows < 0) | (rows >= len(frame)))
            or set(rows.tolist()) != set(query_rows.tolist())
            or not np.array_equal(ids, all_ids[rows])
            or not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 100.0))):
        raise PilotError(f"E{origin} mapped R06 OOF cache identity or values failed")
    cycles = all_cycles[rows]
    index = {int(row): offset for offset, row in enumerate(rows)}
    aligned_prediction = np.asarray([prediction[index[int(row)]] for row in query_rows], dtype=np.float64)
    return aligned_prediction, {"path": OOF_CACHE_REL[origin].as_posix(), "sha256": _sha_file(path),
                        "rows": len(rows), "record_ids_sha256": _sha_array(ids),
                        "cycles_sha256": _sha_array(cycles), "prediction_sha256": _sha_array(prediction)}


def _load_origin(source_root: Path, frame: pd.DataFrame, origin: int) -> OriginInputs:
    fit_rows, query_rows = _prefix_rows(frame, origin)
    all_ids = frame.record_id.astype(str).to_numpy(dtype=str)
    all_cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    docs, all_missing = _text_values(frame.signal_stream)
    text_vectorizer, text_audit = _verify_text_state(source_root, origin, frame, fit_rows)
    position_vectorizer_audit = _verify_positional_vectorizer(source_root, origin, text_vectorizer,
                                                              fit_rows, frame)
    cache_root = _contained_file(source_root, POSITION_RUN_REL)
    manifest_path = _contained_file(cache_root, f"MATRIX_MANIFEST_E{origin}.json")
    manifest = _read_json(manifest_path)
    source_freeze = _read_json(_contained_file(cache_root, "FREEZE.json"))
    freeze_body = {key: value for key, value in source_freeze.items() if key != "freeze_signature"}
    if (_canonical_sha(freeze_body) != source_freeze.get("freeze_signature")
            or source_freeze.get("freeze_signature") != POSITION_FREEZE_SIGNATURE
            or source_freeze.get("input_sha256") != POSITION_INPUT_SHA256
            or manifest.get("origin") != origin
            or manifest.get("freeze_signature") != source_freeze.get("freeze_signature")
            or manifest.get("input_sha256") != source_freeze.get("input_sha256")):
        raise PilotError(f"E{origin} original positional cache freeze linkage failed")
    identity_path = _contained_file(cache_root, f"E{origin}/predictions_E{origin}.npz")
    with np.load(identity_path, allow_pickle=False) as archive:
        # Only target-free row identity members are requested; probability members are never read.
        fit_identity = np.asarray(archive["fit_rows"], dtype=np.int64)
        query_identity = np.asarray(archive["score_rows"], dtype=np.int64)
        query_ids = np.asarray(archive["score_record_ids"], dtype=str)
        query_cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
        present_mask = np.asarray(archive["present_mask"], dtype=bool)
        manifest_digest = str(np.asarray(archive["matrix_manifest_sha256"]).reshape(-1)[0])
    fit_ids = all_ids[fit_rows]
    fit_ids_for_manifest = frame.iloc[fit_rows].record_id.astype(str).to_numpy()
    query_ids_for_manifest = frame.iloc[query_rows].record_id.astype(str).to_numpy()
    expected_fit_meta = manifest.get("fit_row_metadata", {})
    expected_query_meta = manifest.get("score_row_metadata", {})
    if (not np.array_equal(fit_identity, fit_rows) or not np.array_equal(query_identity, query_rows)
            or not np.array_equal(query_ids, all_ids[query_rows])
            or not np.array_equal(query_cycles, all_cycles[query_rows])
            or not np.array_equal(present_mask, ~all_missing[query_rows])
            or manifest_digest != _sha_file(manifest_path)
            or expected_fit_meta.get("rows") != len(fit_rows)
            or expected_fit_meta.get("row_indices_sha256") != _position_sha_array(fit_rows)
            or expected_fit_meta.get("record_ids_sha256") != _position_sha_array(fit_ids_for_manifest)
            or expected_fit_meta.get("cycles_sha256") != _position_sha_array(all_cycles[fit_rows])
            or expected_query_meta.get("rows") != len(query_rows)
            or expected_query_meta.get("row_indices_sha256") != _position_sha_array(query_rows)
            or expected_query_meta.get("record_ids_sha256") != _position_sha_array(query_ids_for_manifest)
            or expected_query_meta.get("cycles_sha256") != _position_sha_array(all_cycles[query_rows])):
        raise PilotError(f"E{origin} positional cache row/ID/cycle/missing identity failed")
    matrix_keys = ("fit_base", "score_base", "fit_position", "score_position",
                   "fit_shuffled_position", "score_shuffled_position")
    matrices = _load_matrix_set(source_root, manifest, matrix_keys)
    fit_base, score_base = matrices["fit_base"], matrices["score_base"]
    fit_real_full, score_real_full = matrices["fit_position"], matrices["score_position"]
    fit_shuffle_full, score_shuffle_full = matrices["fit_shuffled_position"], matrices["score_shuffled_position"]
    width = fit_base.shape[1]
    if (score_base.shape[1] != width
            or not _same_csr(fit_real_full[:, :width], fit_base)
            or not _same_csr(score_real_full[:, :width], score_base)
            or not _same_csr(fit_shuffle_full[:, :width], fit_base)
            or not _same_csr(score_shuffle_full[:, :width], score_base)):
        raise PilotError(f"E{origin} position cache base-prefix invariance failed")
    fit_real = sparse.csr_matrix(fit_real_full[:, width:], dtype=np.float64)
    query_real = sparse.csr_matrix(score_real_full[:, width:], dtype=np.float64)
    fit_shuffled = sparse.csr_matrix(fit_shuffle_full[:, width:], dtype=np.float64)
    query_shuffled = sparse.csr_matrix(score_shuffle_full[:, width:], dtype=np.float64)
    for label, matrix, missing in (
        ("fit real", fit_real, all_missing[fit_rows]), ("query real", query_real, all_missing[query_rows]),
        ("fit shuffled", fit_shuffled, all_missing[fit_rows]), ("query shuffled", query_shuffled, all_missing[query_rows]),
    ):
        if not np.isfinite(matrix.data).all() or matrix[missing].nnz:
            raise PilotError(f"E{origin} {label} cache is nonfinite or nonzero on missing-text rows")
    if (fit_real.shape != fit_shuffled.shape or query_real.shape != query_shuffled.shape
            or fit_real.shape[1] % 3):
        raise PilotError(f"E{origin} REAL/SHUFFLED positional cache dimensions differ")
    fit_bag = unigram_bag(text_vectorizer, [docs[i] for i in fit_rows])
    query_bag = unigram_bag(text_vectorizer, [docs[i] for i in query_rows])
    if fit_bag.shape[1] * 3 != fit_real.shape[1] or query_bag.shape[1] != fit_bag.shape[1]:
        raise PilotError(f"E{origin} transform-only unigram bag does not match positional unigram vocabulary")
    reconstruction = {
        "fit_real": _sample_position_reconstruction(frame, fit_rows, fit_real, text_vectorizer,
                                                      shuffled=False, seed=4200 + origin),
        "fit_shuffled": _sample_position_reconstruction(frame, fit_rows, fit_shuffled, text_vectorizer,
                                                          shuffled=True, seed=4200 + origin),
        "query_real": _sample_position_reconstruction(frame, query_rows, query_real, text_vectorizer,
                                                       shuffled=False, seed=5200 + origin),
        "query_shuffled": _sample_position_reconstruction(frame, query_rows, query_shuffled, text_vectorizer,
                                                           shuffled=True, seed=5200 + origin),
    }
    baseline, baseline_audit = _read_oof_cache(source_root, frame, origin, query_rows)
    fit_frame = frame.iloc[fit_rows]
    query_frame = frame.iloc[query_rows]
    group_features: dict[str, np.ndarray] = {}
    edges_out: dict[str, list[float]] = {}
    for name in ("impact_load", "text_load"):
        prefix = pd.to_numeric(fit_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        query = pd.to_numeric(query_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(prefix).all() or not np.isfinite(query).all():
            raise PilotError(f"E{origin} {name} must be finite for frozen fit-only quartiles")
        edges = np.unique(np.quantile(prefix, [0.25, 0.5, 0.75], method="linear")).astype(float)
        group_features[name] = np.digitize(query, edges, right=False).astype(np.int8)
        edges_out[name] = edges.tolist()
    friction = pd.to_numeric(query_frame.entry_friction, errors="coerce").to_numpy(dtype=np.float64)
    group_features["entry_friction"] = friction
    cache_audit = {
        "manifest_sha256": _sha_file(manifest_path),
        "identity_bundle_sha256": _sha_file(identity_path),
        "matrix_manifest_signature": manifest["freeze_signature"],
        "matrices": {key: {"path": manifest["matrices"][key]["path"],
                            "sha256": manifest["matrices"][key]["sha256"],
                            "logical_sha256": manifest["matrices"][key]["logical_sha256"],
                            "shape": manifest["matrices"][key]["shape"]}
                     for key in matrix_keys},
        "fit_base_width": width, "position_width": fit_real.shape[1],
        "unigram_width": fit_bag.shape[1], "reconstruction": reconstruction,
        "record_ids_fit_sha256": _sha_array(fit_ids), "record_ids_query_sha256": _sha_array(query_ids),
        "fit_rows_sha256_header_aware": _sha_array(fit_rows),
        "query_rows_sha256_header_aware": _sha_array(query_rows),
        "vectorizer": text_audit, "positional_vectorizer_compatibility": position_vectorizer_audit,
        "baseline": baseline_audit,
    }
    return OriginInputs(
        origin=origin, fit_rows=fit_rows, query_rows=query_rows,
        fit_ids=fit_ids, query_ids=query_ids,
        fit_cycles=all_cycles[fit_rows], query_cycles=all_cycles[query_rows],
        fit_missing=all_missing[fit_rows], query_missing=all_missing[query_rows],
        fit_bag=fit_bag, query_bag=query_bag,
        fit_position=fit_real, query_position=query_real,
        fit_shuffled=fit_shuffled, query_shuffled=query_shuffled,
        fit_baseline=np.full(len(fit_rows), np.nan, dtype=np.float64),
        query_baseline=baseline, group_features=group_features,
        group_edges=edges_out, audit=cache_audit,
    )


def _source_hashes(source_root: Path) -> dict[str, str]:
    source_files: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = _contained_file(ROOT, relative)
        source_files[relative] = _sha_file(path)
    source_files[TRAIN_REL.as_posix()] = _sha_file(_contained_file(source_root, TRAIN_REL))
    source_files[CLASSIFICATION_TRAIN_REL.as_posix()] = _sha_file(
        _contained_file(source_root, CLASSIFICATION_TRAIN_REL))
    for origin in ORIGINS:
        posroot = _contained_file(source_root, POSITION_RUN_REL)
        textdir = _contained_file(source_root, TEXT_RUN_REL / f"E{origin}")
        oof = _contained_file(source_root, OOF_CACHE_REL[origin])
        rels = [
            (POSITION_RUN_REL / "FREEZE.json"),
            POSITION_BUDGET_REL,
            (POSITION_RUN_REL / f"MATRIX_MANIFEST_E{origin}.json"),
            (POSITION_RUN_REL / f"E{origin}/predictions_E{origin}.npz"),
            (POSITION_RUN_REL / f"receipts/POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF.json"),
            (POSITION_RUN_REL / f"units/POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF.joblib"),
            (TEXT_RUN_REL / f"E{origin}/receipt.json"),
            (TEXT_RUN_REL / f"E{origin}/text_state.json"),
            (TEXT_RUN_REL / f"E{origin}/vectorizer.joblib"),
            OOF_CACHE_REL[origin],
        ]
        for name in ("fit_base", "score_base", "fit_position", "score_position",
                     "fit_shuffled_position", "score_shuffled_position"):
            rels.append(Path(_read_json(posroot / f"MATRIX_MANIFEST_E{origin}.json")["matrices"][name]["path"]))
        for relative in rels:
            path = _contained_file(source_root, relative)
            source_files[relative.as_posix()] = _sha_file(path)
    return source_files


def _freeze_body(source_root: Path, frame: pd.DataFrame,
                 origins: Mapping[int, OriginInputs]) -> dict[str, Any]:
    recipe = _contained_file(ROOT, RECIPE_REL)
    source_status = _git(source_root, "status", "--porcelain")
    source_pair_alignment = _verify_classification_regression_alignment(source_root, frame)
    return {
        "schema_version": 1,
        "card_id": CARD_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "source_git_head": _git(source_root, "rev-parse", "HEAD"),
        "source_git_branch": _git(source_root, "branch", "--show-current"),
        "source_worktree_status_sha256": _canonical_sha(source_status.splitlines()),
        "local_root": str(ROOT.resolve()),
        "local_git_head": _git(ROOT, "rev-parse", "HEAD"),
        "local_git_branch": _git(ROOT, "branch", "--show-current"),
        "official_train_sha256": TRAIN_SHA256,
        "feature_columns": list(FEATURE_COLUMNS),
        "classification_regression_feature_alignment": source_pair_alignment,
        "recipe_sha256": _sha_file(recipe),
        "source_hashes": _source_hashes(source_root),
        "resolved_parameters": _resolved_parameters(),
        "fit_order": list(FIT_IDS),
        "fit_cap": 12,
        "resource_gate": {"minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
                           "reserve_bytes": MIN_FREE_RAM_RESERVE, "max_process_rss_bytes": MAX_RSS,
                           "threads": THREADS, "jobs": 1},
        "targets_read": False,
        "origins": {
            str(origin): {
                "fit_rows": len(item.fit_rows), "query_rows": len(item.query_rows),
                "fit_cycle_bounds": [int(item.fit_cycles.min()), int(item.fit_cycles.max())],
                "query_cycle_bounds": [int(item.query_cycles.min()), int(item.query_cycles.max())],
                "fit_row_indices_sha256": _sha_array(item.fit_rows),
                "query_row_indices_sha256": _sha_array(item.query_rows),
                "fit_record_ids_sha256": _sha_array(item.fit_ids),
                "query_record_ids_sha256": _sha_array(item.query_ids),
                "fit_cycles_sha256": _sha_array(item.fit_cycles),
                "query_cycles_sha256": _sha_array(item.query_cycles),
                "missing_fit_rows": int(item.fit_missing.sum()),
                "missing_query_rows": int(item.query_missing.sum()),
                "group_edges": item.group_edges,
                **item.audit,
            }
            for origin, item in origins.items()
        },
    }


def preflight(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    source_root = source_root.resolve()
    frame = _load_feature_frame(source_root)
    origins = {origin: _load_origin(source_root, frame, origin) for origin in ORIGINS}
    body = _freeze_body(source_root, frame, origins)
    existing = _read_json(FREEZE_PATH) if FREEZE_PATH.is_file() else None
    if existing:
        old_body = {key: value for key, value in existing.items() if key != "freeze_signature"}
        if existing.get("freeze_signature") != _canonical_sha(old_body):
            raise PilotError("Existing pilot FREEZE.json signature failed")
        for key in ("source_root", "source_git_head", "official_train_sha256", "recipe_sha256",
                    "source_hashes", "resolved_parameters", "fit_order", "origins"):
            if existing.get(key) != body.get(key):
                raise PilotError(f"Frozen pilot source changed after preflight: {key}")
        freeze = existing
    else:
        if SUPERSEDED_FREEZE_PATH.is_file():
            previous = _read_json(SUPERSEDED_FREEZE_PATH)
            previous_body = {key: value for key, value in previous.items() if key != "freeze_signature"}
            if previous.get("freeze_signature") != _canonical_sha(previous_body):
                raise PilotError("Preserved provisional FREEZE.json signature failed")
            events = _read_events(LEDGER_PATH)
            if any(event.get("event") == "FIT_STARTED" for event in events):
                raise PilotError("Cannot supersede a freeze after any production fit start")
            body["supersedes_prefit_freeze"] = {
                "path": (SUPERSEDED_FREEZE_PATH.relative_to(ROOT)).as_posix(),
                "sha256": _sha_file(SUPERSEDED_FREEZE_PATH),
                "freeze_signature": previous["freeze_signature"],
                "production_fit_starts": 0,
                "reason": "Prefit implementation corrections; no production fit unit had started.",
            }
        body["freeze_signature"] = _canonical_sha(body)
        freeze = body
        _atomic_json(FREEZE_PATH, freeze)
    memory = psutil.virtual_memory()
    audit = {
        "schema_version": 1, "card_id": CARD_ID, "status": "PASS",
        "freeze_signature": freeze["freeze_signature"],
        "query_targets_read": False, "official_target_column_loaded": False,
        "source_or_cache_mutation": False,
        "classification_regression_feature_alignment": freeze["classification_regression_feature_alignment"],
        "origin_rows": {str(o): {"fit_rows": len(v.fit_rows), "query_rows": len(v.query_rows),
                                 "fit_cycle_bounds": [int(v.fit_cycles.min()), int(v.fit_cycles.max())],
                                 "query_cycle_bounds": [int(v.query_cycles.min()), int(v.query_cycles.max())],
                                 "fit_missing": int(v.fit_missing.sum()), "query_missing": int(v.query_missing.sum()),
                                 "unigram_width": v.fit_bag.shape[1], "position_width": v.fit_position.shape[1],
                                 "reconstruction_samples": v.audit["reconstruction"]}
                       for o, v in origins.items()},
        "vectorizer_position_compatibility": {
            str(o): {"numeric_state_equal_to_standalone": v.audit["positional_vectorizer_compatibility"]["numeric_state_equal_to_standalone"],
                     "vocabulary_dict_equal": v.audit["positional_vectorizer_compatibility"]["vocabulary_dict_equal"],
                     "parameters_equal": v.audit["positional_vectorizer_compatibility"]["parameters_equal"],
                     "max_abs_idf_difference": v.audit["positional_vectorizer_compatibility"]["max_abs_idf_difference"]}
            for o, v in origins.items()
        },
        "resources_before_fit": {"available_ram_bytes": int(memory.available),
                                  "process_rss_bytes": int(psutil.Process().memory_info().rss),
                                  "fit_units_started": 0},
    }
    if memory.available < MIN_AVAILABLE_RAM:
        audit["status"] = "BLOCKED_RESOURCES"
    if "supersedes_prefit_freeze" in freeze:
        audit["supersedes_prefit_freeze"] = freeze["supersedes_prefit_freeze"]
    _atomic_json(PREFIT_RUN_PATH, audit)
    _atomic_json(PREFIT_REPORT_PATH, audit)
    if audit["status"] != "PASS":
        raise PilotError("Prefit RAM gate failed; no production fit started")
    return {"freeze": freeze, "preflight": audit, "origins": origins}



def _read_targets_for_rows(path: Path, rows: np.ndarray, expected_ids: np.ndarray,
                           *, expected_total_rows: int) -> np.ndarray:
    """Decode and parse only selected physical CSV records and their target cells."""
    wanted_rows = np.asarray(rows, dtype=np.int64)
    ids = np.asarray(expected_ids, dtype=str)
    if (wanted_rows.ndim != 1 or ids.shape != wanted_rows.shape or not len(wanted_rows)
            or len(set(wanted_rows.tolist())) != len(wanted_rows)
            or not np.array_equal(wanted_rows, np.sort(wanted_rows))
            or np.any((wanted_rows < 0) | (wanted_rows >= expected_total_rows))):
        raise PilotError("Requested fit target rows and IDs must be unique aligned vectors")
    wanted = set(int(row) for row in wanted_rows)
    values = np.empty(len(wanted_rows), dtype=np.float64)
    observed_ids: list[str] = []
    parsed = 0
    physical_lines = 0
    with path.open("rb") as stream:
        for physical_line, raw_line in enumerate(stream):
            physical_lines += 1
            if physical_line == 0:
                try:
                    line = raw_line.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise PilotError("CSV header is not valid UTF-8") from exc
                try:
                    header_rows = list(csv.reader([line], strict=True))
                except csv.Error as exc:
                    raise PilotError("Regression Train header is not valid single-line CSV") from exc
                if len(header_rows) != 1 or len(set(header_rows[0])) != len(header_rows[0]):
                    raise PilotError("Regression Train header is empty or has duplicate columns")
                header = header_rows[0]
                if "record_id" not in header or "threat_rank" not in header:
                    raise PilotError("Regression Train lacks required identity or target columns")
                id_column = header.index("record_id")
                target_column = header.index("threat_rank")
                continue

            row_index = physical_line - 1
            if row_index not in wanted:
                # Do not decode, tokenize, or convert unselected source rows.
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PilotError(f"Selected source row {row_index} is not valid UTF-8") from exc
            try:
                row_records = list(csv.reader([line], strict=True))
            except csv.Error as exc:
                raise PilotError(f"Selected source row {row_index} is not valid single-line CSV") from exc
            if len(row_records) != 1 or len(row_records[0]) != len(header):
                raise PilotError(f"Selected source row {row_index} has an unexpected CSV shape")
            fields = row_records[0]
            observed_ids.append(fields[id_column])
            try:
                values[parsed] = float(fields[target_column])
            except (TypeError, ValueError) as exc:
                raise PilotError(f"Selected source row {row_index} has a non-numeric threat_rank") from exc
            parsed += 1

    if physical_lines != expected_total_rows + 1:
        raise PilotError("Regression Train physical-line count differs from feature row count")
    if parsed != len(wanted_rows):
        raise PilotError("Selected target parser did not return every requested source row")
    if not np.array_equal(np.asarray(observed_ids, dtype=str), ids):
        raise PilotError("Fit target row IDs differ from the frozen selected source rows")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise PilotError("Selected same-origin threat_rank values must be finite and within [0,100]")
    return values


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except Exception as exc:
            raise PilotError(f"Invalid fit ledger line {line_no}") from exc
        if not isinstance(item, dict):
            raise PilotError(f"Invalid fit ledger event at line {line_no}")
        events.append(item)
    return events


def _append_event(event: Mapping[str, Any], path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical(dict(event)) + b"\n"
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _resource_snapshot() -> dict[str, Any]:
    return {"available_ram_bytes": int(psutil.virtual_memory().available),
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
            "max_rss_bytes": MAX_RSS, "threads": THREADS, "jobs": 1}


def _active_known_training_processes() -> list[dict[str, Any]]:
    current_pid = os.getpid()
    candidates: list[tuple[psutil.Process, str, float]] = []
    needles = ("lightgbm", "reg_position_direct_l1", "reg_position_l1_v1",
               "optimization_v3_positional", "regression_research")
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        if proc.info.get("pid") == current_pid or "python" not in str(proc.info.get("name", "")).lower():
            continue
        try:
            command = " ".join(proc.cmdline()).lower()
            if any(needle in command for needle in needles):
                cpu = proc.cpu_times()
                candidates.append((proc, command[:300], float(cpu.user + cpu.system)))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    time.sleep(0.25)
    active: list[dict[str, Any]] = []
    for proc, command, before in candidates:
        try:
            cpu = proc.cpu_times()
            after = float(cpu.user + cpu.system)
            if after - before >= 0.05:
                active.append({"pid": proc.pid, "command": command, "cpu_delta_seconds": after - before})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return active


def _resource_gate(estimated_bytes: int) -> dict[str, Any]:
    snapshot = _resource_snapshot()
    active = _active_known_training_processes()
    if active:
        raise PilotError(f"Concurrent KASDAD training compute detected; callback root before fitting: {active}")
    if (snapshot["available_ram_bytes"] < MIN_AVAILABLE_RAM
            or snapshot["available_ram_bytes"] - estimated_bytes < MIN_FREE_RAM_RESERVE
            or snapshot["process_rss_bytes"] + estimated_bytes > MAX_RSS):
        raise PilotError(f"Resource gate failed before fit unit: {snapshot}; estimate={estimated_bytes}")
    return {**snapshot, "estimated_incremental_memory_bytes": int(estimated_bytes),
            "minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
            "minimum_free_ram_reserve_bytes": MIN_FREE_RAM_RESERVE,
            "active_training_processes": active}


class _LedgerFileLock:
    """Acquire an OS lock on a persistent file; never delete the lock file."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        import msvcrt
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise PilotError("Another REG_POSITION_DIRECT_L1_V1 fit process holds the run lock") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        import msvcrt
        if self.stream is not None:
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            self.stream.close()
            self.stream = None


def _unit_paths(fit_id: str, output_root: Path = RUN_ROOT) -> tuple[Path, Path]:
    return output_root / "units" / f"{fit_id}.joblib", output_root / "receipts" / f"{fit_id}.json"


def _prediction_paths(fit_id: str, output_root: Path = RUN_ROOT) -> tuple[Path, Path]:
    return output_root / "predictions" / f"{fit_id}.npz", output_root / "predictions" / f"{fit_id}.json"


def _receipt_body(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def _validate_unit(output_root: Path, fit_id: str, freeze: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    artifact = output_root / "units" / f"{fit_id}.joblib"
    receipt_path = output_root / "receipts" / f"{fit_id}.json"
    if not artifact.is_file() or not receipt_path.is_file():
        raise PilotError(f"Durable unit is missing artifact or receipt: {fit_id}")
    receipt = _read_json(receipt_path)
    if (receipt.get("schema_version") != 1 or receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze.get("freeze_signature")
            or receipt.get("artifact_sha256") != _sha_file(artifact)
            or receipt.get("receipt_sha256") != _canonical_sha(_receipt_body(receipt))):
        raise PilotError(f"Durable fit unit receipt/hash/schema failed: {fit_id}")
    events = _read_events(output_root / "fit_ledger.jsonl")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    saves = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id]
    fails = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == fit_id]
    if (len(starts) != 1 or len(saves) != 1 or fails
            or saves[0].get("receipt_sha256") != _sha_file(receipt_path)):
        raise PilotError(f"Fit ledger does not prove one start and one durable save: {fit_id}")
    return joblib.load(artifact), receipt


def _fit_unit(fit_id: str, origin: int, component: str, freeze: Mapping[str, Any],
              fit_callable: Callable[[], tuple[Any, Mapping[str, Any]]], *,
              estimated_bytes: int, synthetic: bool = False,
              output_root: Path = RUN_ROOT) -> tuple[Any, dict[str, Any], bool]:
    artifact, receipt_path = _unit_paths(fit_id, output_root)
    output_root = artifact.parents[1]
    events = _read_events(output_root / "fit_ledger.jsonl")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in ("FIT_SAVED", "FIT_FAILED") and e.get("fit_id") == fit_id]
    if artifact.exists() or receipt_path.exists() or starts or terminal:
        if starts and not terminal:
            raise PilotError(f"Consumed fit unit has no durable result; retries prohibited: {fit_id}")
        value, receipt = _validate_unit(output_root, fit_id, freeze)
        return value, receipt, False
    prior_ids = [e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"]
    expected_index = len(prior_ids)
    if expected_index >= len(FIT_IDS) or FIT_IDS[expected_index] != fit_id:
        raise PilotError(f"Fit order/cap violation: next={FIT_IDS[expected_index] if expected_index < len(FIT_IDS) else 'CAP'}, got={fit_id}")
    resource_before = {} if synthetic else _resource_gate(estimated_bytes)
    start = {"event": "FIT_STARTED", "fit_id": fit_id, "origin": origin,
             "component": component, "freeze_signature": freeze["freeze_signature"],
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "resource_before": resource_before, "synthetic": synthetic}
    _append_event(start, output_root / "fit_ledger.jsonl")
    started = time.perf_counter()
    try:
        value, metadata = fit_callable()
        if not synthetic:
            after_fit = _resource_snapshot()
            if (after_fit["process_rss_bytes"] > MAX_RSS
                    or after_fit["available_ram_bytes"] < MIN_FREE_RAM_RESERVE):
                raise PilotError(f"Resource cap/reserve exceeded after consumed fit unit: {after_fit}")
        _atomic_joblib(artifact, value)
        receipt = {
            "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id,
            "origin": origin, "component": component,
            "freeze_signature": freeze["freeze_signature"],
            "artifact_sha256": _sha_file(artifact),
            "fit_elapsed_seconds": time.perf_counter() - started,
            "fit_metadata": dict(metadata), "synthetic": synthetic,
        }
        receipt["receipt_sha256"] = _canonical_sha(receipt)
        _atomic_json(receipt_path, receipt)
        receipt_sha = _sha_file(receipt_path)
        _append_event({"event": "FIT_SAVED", "fit_id": fit_id, "origin": origin,
                       "component": component, "freeze_signature": freeze["freeze_signature"],
                       "receipt_sha256": receipt_sha,
                       "saved_utc": datetime.now(timezone.utc).isoformat(),
                       "resource_after": {} if synthetic else _resource_snapshot(),
                       "synthetic": synthetic}, output_root / "fit_ledger.jsonl")
        return value, receipt, True
    except Exception as exc:
        _append_event({"event": "FIT_FAILED", "fit_id": fit_id, "origin": origin,
                       "component": component, "freeze_signature": freeze["freeze_signature"],
                       "failed_utc": datetime.now(timezone.utc).isoformat(),
                       "error": f"{type(exc).__name__}: {exc}", "synthetic": synthetic},
                      output_root / "fit_ledger.jsonl")
        raise


def _structured_tokens(frame: pd.DataFrame) -> np.ndarray:
    missing_marker = "<TP2_MISSING_CATEGORY>"
    raw = {name: np.empty(len(frame), dtype=object) for name in CATEGORICAL_FEATURES}
    for name in CATEGORICAL_FEATURES:
        for index, value in enumerate(frame[name].to_numpy(dtype=object)):
            raw[name][index] = missing_marker if _is_missing(value) else "V:" + str(value).strip()
    return np.column_stack([raw[name] for name in CATEGORICAL_FEATURES])


def _fit_or_transform_structured(fit_id_prefix: str, origin: int, freeze: Mapping[str, Any],
                                 fit_frame: pd.DataFrame, query_frame: pd.DataFrame,
                                 *, synthetic: bool = False,
                                 output_root: Path = RUN_ROOT) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    fit_struct = fit_frame.loc[:, list(STRUCTURED_FEATURES)]
    query_struct = query_frame.loc[:, list(STRUCTURED_FEATURES)]
    processor_id = f"{fit_id_prefix}_STRUCTURED"

    def fit_processor():
        processor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        processed, _ = processor.fit_transform(fit_struct)
        names = processor.transformed_feature_names
        categorical = list(processor.categorical_positions)
        numeric = [i for i in range(len(names)) if i not in set(categorical)]
        return {"processor": processor, "numeric_positions": numeric,
                "feature_names": list(names), "categorical_positions": categorical}, {
                    "underlying_fit_calls": ["_SelectedPreprocessor.fit_transform x1 (one nested OrdinalEncoder.fit)"],
                    "fit_rows": len(fit_struct), "transformed_width": int(processed.shape[1]),
                    "feature_names_sha256": _canonical_sha(list(names)),
                    "numeric_imputer_medians": dict(processor.medians),
                    "numeric_imputer_medians_sha256": _canonical_sha(processor.medians),
                    "category_counts": processor.category_counts,
                    "category_counts_sha256": _canonical_sha(processor.category_counts),
                }

    processor_state, processor_receipt, processor_new = _fit_unit(
        processor_id, origin, "STRUCTURED", freeze, fit_processor,
        estimated_bytes=512 * 1024**2, synthetic=synthetic, output_root=output_root)
    processor = processor_state["processor"]
    numeric_positions = list(processor_state["numeric_positions"])
    fit_processed = processor.transform(fit_struct)[0] if not processor_new else None
    if processor_new:
        # Recover the training transform from the saved processor to keep the same path on resume.
        fit_processed = processor.transform(fit_struct)[0]
    query_processed, _ = processor.transform(query_struct)
    fit_numeric = np.asarray(fit_processed[:, numeric_positions], dtype=np.float64)
    query_numeric = np.asarray(query_processed[:, numeric_positions], dtype=np.float64)

    scaler_id = f"{fit_id_prefix}_SCALER"

    def fit_scaler():
        scaler = StandardScaler()
        scaler.fit(fit_numeric)
        return scaler, {"underlying_fit_calls": ["StandardScaler.fit x1"],
                        "fit_rows": len(fit_numeric), "feature_count": int(fit_numeric.shape[1]),
                        "mean_sha256": _sha_array(scaler.mean_), "scale_sha256": _sha_array(scaler.scale_)}

    scaler, _, _ = _fit_unit(scaler_id, origin, "SCALER", freeze, fit_scaler,
                             estimated_bytes=256 * 1024**2, synthetic=synthetic, output_root=output_root)
    fit_numeric_scaled = scaler.transform(fit_numeric)
    query_numeric_scaled = scaler.transform(query_numeric)

    fit_tokens = _structured_tokens(fit_frame)
    query_tokens = _structured_tokens(query_frame)
    ohe_id = f"{fit_id_prefix}_OHE"

    def fit_ohe():
        encoder = OneHotEncoder(min_frequency=30, max_categories=254,
                                handle_unknown="infrequent_if_exist", sparse_output=True,
                                dtype=np.float64)
        encoder.fit(fit_tokens)
        vocab_sha = _canonical_sha([[str(v) for v in values] for values in encoder.categories_])
        return encoder, {"underlying_fit_calls": ["OneHotEncoder.fit x1"],
                         "fit_rows": len(fit_tokens), "output_width": len(encoder.get_feature_names_out()),
                         "categories_sha256": vocab_sha}

    encoder, _, _ = _fit_unit(ohe_id, origin, "OHE", freeze, fit_ohe,
                              estimated_bytes=384 * 1024**2, synthetic=synthetic, output_root=output_root)
    fit_categorical = encoder.transform(fit_tokens)
    query_categorical = encoder.transform(query_tokens)
    fit_structured = sparse.hstack((sparse.csr_matrix(fit_numeric_scaled), fit_categorical),
                                   format="csr", dtype=np.float64)
    query_structured = sparse.hstack((sparse.csr_matrix(query_numeric_scaled), query_categorical),
                                     format="csr", dtype=np.float64)
    if (not np.isfinite(fit_structured.data).all() or not np.isfinite(query_structured.data).all()
            or fit_structured.shape[1] != query_structured.shape[1]):
        raise PilotError("Structured fit/query transforms are invalid or have different widths")
    return fit_structured, query_structured, {
        "processor_receipt_sha256": _sha_file(_unit_paths(processor_id, output_root)[1]),
        "scaler_receipt_sha256": _sha_file(_unit_paths(scaler_id, output_root)[1]),
        "ohe_receipt_sha256": _sha_file(_unit_paths(ohe_id, output_root)[1]),
        "fit_shape": list(fit_structured.shape), "query_shape": list(query_structured.shape),
    }


def _rank50(raw: Any, cycles: Any) -> np.ndarray:
    raw_arr = np.asarray(raw, dtype=np.float64)
    cycle_arr = np.asarray(cycles, dtype=np.int64)
    if (raw_arr.ndim != 1 or cycle_arr.shape != raw_arr.shape or not len(raw_arr)
            or not np.isfinite(raw_arr).all()):
        raise PilotError("Rank50 requires nonempty aligned finite raw predictions and cycles")
    output = np.empty(len(raw_arr), dtype=np.float64)
    for cycle in np.unique(cycle_arr):
        mask = cycle_arr == cycle
        percentile = 100.0 * rankdata(raw_arr[mask], method="average") / int(mask.sum())
        output[mask] = 0.5 * raw_arr[mask] + 0.5 * percentile
    if not np.isfinite(output).all():
        raise PilotError("Rank50 mapping produced nonfinite output")
    return output


def _combine_design(structured: sparse.csr_matrix, bag: sparse.csr_matrix,
                    position: sparse.csr_matrix | None, missing: np.ndarray) -> sparse.csr_matrix:
    mask = np.asarray(missing, dtype=bool)
    if structured.shape[0] != bag.shape[0] or mask.shape != (structured.shape[0],):
        raise PilotError("Structured, bag, and missing indicator rows are not aligned")
    blocks: list[sparse.spmatrix] = [structured, bag]
    if position is not None:
        if position.shape[0] != structured.shape[0]:
            raise PilotError("Position suffix rows are not aligned")
        blocks.append(position)
    blocks.append(sparse.csr_matrix(mask.astype(np.float64).reshape(-1, 1)))
    design = sparse.hstack(blocks, format="csr", dtype=np.float64)
    design.sum_duplicates(); design.sort_indices()
    if not np.isfinite(design.data).all():
        raise PilotError("Arm design contains nonfinite values")
    return design


def _validate_prediction_bundle(path: Path, receipt_path: Path, *, freeze: Mapping[str, Any],
                                fit_id: str, origin: int, arm: str,
                                query_rows: np.ndarray, query_ids: np.ndarray,
                                query_cycles: np.ndarray, missing: np.ndarray,
                                baseline: np.ndarray) -> dict[str, np.ndarray]:
    if not path.is_file() or not receipt_path.is_file():
        raise PilotError(f"Missing prediction bundle/receipt for {fit_id}")
    receipt = _read_json(receipt_path)
    if (receipt.get("schema_version") != 1 or receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or receipt.get("bundle_sha256") != _sha_file(path)
            or receipt.get("receipt_sha256") != _canonical_sha(_receipt_body(receipt))):
        raise PilotError(f"Prediction bundle receipt failed for {fit_id}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in (
            "origin", "arm", "fit_rows", "query_rows", "query_record_ids", "query_cycles",
            "missing_mask", "baseline", "raw_auxiliary", "mapped_auxiliary", "prediction")}
    raw = np.asarray(arrays["raw_auxiliary"], dtype=np.float64)
    mapped = np.asarray(arrays["mapped_auxiliary"], dtype=np.float64)
    pred = np.asarray(arrays["prediction"], dtype=np.float64)
    if (int(np.asarray(arrays["origin"]).reshape(-1)[0]) != origin
            or str(np.asarray(arrays["arm"]).reshape(-1)[0]) != arm
            or not np.array_equal(np.asarray(arrays["query_rows"], dtype=np.int64), query_rows)
            or not np.array_equal(np.asarray(arrays["query_record_ids"], dtype=str), query_ids)
            or not np.array_equal(np.asarray(arrays["query_cycles"], dtype=np.int64), query_cycles)
            or not np.array_equal(np.asarray(arrays["missing_mask"], dtype=bool), missing)
            or not np.array_equal(np.asarray(arrays["baseline"], dtype=np.float64), baseline)
            or raw.shape != baseline.shape or mapped.shape != baseline.shape or pred.shape != baseline.shape
            or not np.isfinite(raw).all() or not np.isfinite(mapped).all() or not np.isfinite(pred).all()):
        raise PilotError(f"Prediction IDs/shapes/values failed for {fit_id}")
    expected_mapped = _rank50(raw, query_cycles)
    expected = np.clip(0.75 * baseline + 0.25 * expected_mapped, 0.0, 100.0)
    expected[missing] = baseline[missing]
    if (not np.array_equal(mapped, expected_mapped) or not np.array_equal(pred, expected)
            or not np.array_equal(pred[missing], baseline[missing])
            or np.any((pred < 0.0) | (pred > 100.0))):
        raise PilotError(f"Prediction formula or exact missing fallback failed for {fit_id}")
    return {"raw_auxiliary": raw, "mapped_auxiliary": mapped, "prediction": pred,
            "baseline": np.asarray(arrays["baseline"], dtype=np.float64)}


def _save_prediction(fit_id: str, origin: int, arm: str, freeze: Mapping[str, Any],
                     model_receipt: Mapping[str, Any], fit_rows: np.ndarray,
                     query_rows: np.ndarray, query_ids: np.ndarray, query_cycles: np.ndarray,
                     missing: np.ndarray, baseline: np.ndarray, raw: np.ndarray,
                     output_root: Path = RUN_ROOT) -> dict[str, np.ndarray]:
    path, receipt_path = _prediction_paths(fit_id, output_root)
    if path.exists() or receipt_path.exists():
        return _validate_prediction_bundle(path, receipt_path, freeze=freeze, fit_id=fit_id,
                                           origin=origin, arm=arm, query_rows=query_rows,
                                           query_ids=query_ids, query_cycles=query_cycles,
                                           missing=missing, baseline=baseline)
    mapped = _rank50(raw, query_cycles)
    pred = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    # Full-cycle ranking and blend are complete before missing rows are replaced by R06.
    pred[missing] = baseline[missing]
    _atomic_npz(path, origin=np.asarray([origin], dtype=np.int64), arm=np.asarray([arm], dtype=str),
                fit_rows=np.asarray(fit_rows, dtype=np.int64),
                query_rows=np.asarray(query_rows, dtype=np.int64),
                query_record_ids=np.asarray(query_ids, dtype=str),
                query_cycles=np.asarray(query_cycles, dtype=np.int64),
                missing_mask=np.asarray(missing, dtype=bool), baseline=np.asarray(baseline, dtype=np.float64),
                raw_auxiliary=np.asarray(raw, dtype=np.float64),
                mapped_auxiliary=np.asarray(mapped, dtype=np.float64), prediction=np.asarray(pred, dtype=np.float64))
    receipt = {"schema_version": 1, "fit_id": fit_id,
               "freeze_signature": freeze["freeze_signature"],
               "bundle_sha256": _sha_file(path), "model_receipt_sha256": model_receipt["receipt_sha256"],
               "query_rows": len(query_rows), "query_ids_sha256": _sha_array(query_ids),
               "query_cycles_sha256": _sha_array(query_cycles), "missing_rows": int(missing.sum()),
               "raw_auxiliary_sha256": _sha_array(raw), "mapped_auxiliary_sha256": _sha_array(mapped),
               "prediction_sha256": _sha_array(pred), "rank_before_missing_mask": True,
               "query_targets_read": False}
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    _atomic_json(receipt_path, receipt)
    return _validate_prediction_bundle(path, receipt_path, freeze=freeze, fit_id=fit_id,
                                       origin=origin, arm=arm, query_rows=query_rows,
                                       query_ids=query_ids, query_cycles=query_cycles,
                                       missing=missing, baseline=baseline)


def _fit_one_origin(source_root: Path, frame: pd.DataFrame, item: OriginInputs,
                    freeze: Mapping[str, Any], *, synthetic: bool = False,
                    interruption_hook: Callable[[str], None] | None = None,
                    model_params_override: Mapping[str, Any] | None = None,
                    synthetic_y: np.ndarray | None = None,
                    output_root: Path = RUN_ROOT) -> dict[str, Any]:
    origin = item.origin
    fit_frame = frame.iloc[item.fit_rows].loc[:, list(HYBRID_FEATURES)].copy()
    query_frame = frame.iloc[item.query_rows].loc[:, list(HYBRID_FEATURES)].copy()
    prefix = f"{CARD_ID}_E{origin}"
    structured_fit, structured_query, structured_meta = _fit_or_transform_structured(
        prefix, origin, freeze, fit_frame, query_frame, synthetic=synthetic, output_root=output_root)
    y_fit = (np.asarray(synthetic_y, dtype=np.float64) if synthetic else
             _read_targets_for_rows(_contained_file(source_root, TRAIN_REL), item.fit_rows, item.fit_ids,
                                    expected_total_rows=len(frame)))
    if y_fit is not None and (len(y_fit) != len(item.fit_rows) or not np.isfinite(y_fit).all()):
        raise PilotError(f"E{origin} direct target failed fit-row alignment")
    fit_missing_col = np.asarray(item.fit_missing, dtype=bool)
    query_missing_col = np.asarray(item.query_missing, dtype=bool)
    raw_predictions: dict[str, np.ndarray] = {}
    for arm in ARMS:
        fit_id = f"{prefix}_{arm}"
        fit_position = None if arm == "BAG" else (item.fit_position if arm == "REAL" else item.fit_shuffled)
        query_position = None if arm == "BAG" else (item.query_position if arm == "REAL" else item.query_shuffled)
        X_fit = _combine_design(structured_fit, item.fit_bag, fit_position, fit_missing_col)
        X_query = _combine_design(structured_query, item.query_bag, query_position, query_missing_col)
        params = dict(model_params_override or _resolved_parameters())

        def fit_model() -> tuple[Any, Mapping[str, Any]]:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
            with threadpool_limits(limits=THREADS):
                model.fit(X_fit, y_fit)
            return model, {"underlying_fit_calls": ["LGBMRegressor.fit x1"],
                           "fit_rows": len(item.fit_rows), "feature_count": int(X_fit.shape[1]),
                           "objective": params.get("objective"), "parameters": params,
                           "fit_target_sha256": _sha_array(y_fit) if y_fit is not None else None,
                           "direct_target": "threat_rank"}

        estimated = int(max(256 * 1024**2, 2.0 * (X_fit.data.nbytes + X_fit.indices.nbytes + X_fit.indptr.nbytes)))
        model, model_receipt, newly_fit = _fit_unit(
            fit_id, origin, arm, freeze, fit_model, estimated_bytes=estimated,
            synthetic=synthetic, output_root=output_root)
        if interruption_hook and newly_fit:
            interruption_hook(fit_id)
        pred_path, pred_receipt = _prediction_paths(fit_id, output_root)
        if pred_path.is_file() and pred_receipt.is_file():
            bundles = _validate_prediction_bundle(pred_path, pred_receipt, freeze=freeze, fit_id=fit_id,
                                                  origin=origin, arm=arm, query_rows=item.query_rows,
                                                  query_ids=item.query_ids, query_cycles=item.query_cycles,
                                                  missing=item.query_missing, baseline=item.query_baseline)
        else:
            with threadpool_limits(limits=THREADS):
                raw = np.asarray(model.predict(X_query), dtype=np.float64)
            if raw.shape != item.query_baseline.shape or not np.isfinite(raw).all():
                raise PilotError(f"E{origin} {arm} raw query predictions are invalid")
            bundles = _save_prediction(fit_id, origin, arm, freeze, model_receipt,
                                       item.fit_rows, item.query_rows, item.query_ids,
                                       item.query_cycles, item.query_missing, item.query_baseline, raw,
                                       output_root=output_root)
        raw_predictions[arm] = bundles["raw_auxiliary"]
        del model, X_fit, X_query
    del structured_fit, structured_query
    return {"origin": origin, "structured": structured_meta,
            "prediction_paths": {arm: _prediction_paths(f"{prefix}_{arm}", output_root)[0].as_posix() for arm in ARMS},
            "prediction_hashes": {arm: _sha_file(_prediction_paths(f"{prefix}_{arm}", output_root)[0]) for arm in ARMS},
            "raw_shapes": {arm: list(raw_predictions[arm].shape) for arm in ARMS}}


def fit_official(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not FREEZE_PATH.is_file():
        raise PilotError("Run preflight before any production fit")
    freeze = _read_json(FREEZE_PATH)
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if freeze.get("freeze_signature") != _canonical_sha(body):
        raise PilotError("Frozen input signature failed")
    if str(source_root) != freeze.get("source_root"):
        raise PilotError("Production source root differs from the exact prefit source path")
    source_identity = {
        "source_git_head": _git(source_root, "rev-parse", "HEAD"),
        "source_git_branch": _git(source_root, "branch", "--show-current"),
        "source_worktree_status_sha256": _canonical_sha(
            _git(source_root, "status", "--porcelain").splitlines()),
    }
    if any(source_identity[key] != freeze.get(key) for key in source_identity):
        raise PilotError("Production source checkout identity/status differs from prefit freeze")
    if _git(ROOT, "rev-parse", "HEAD") != freeze.get("local_git_head"):
        raise PilotError("Local task checkout HEAD changed after prefit freeze")
    for relative, expected in freeze.get("source_hashes", {}).items():
        path = _contained_file(source_root if relative.startswith(("data/", "experiments/")) else ROOT,
                               relative)
        if _sha_file(path) != expected:
            raise PilotError(f"Frozen source file changed after preflight: {relative}")
    synthetic_path = _contained_file(ROOT, REPORT_REL / "SYNTHETIC_E2E.json")
    synthetic = _read_json(synthetic_path)
    if (synthetic.get("status") != "PASS" or synthetic.get("production_fit_units") != 0
            or synthetic.get("learned_fit_call_total") != 12
            or synthetic.get("runner_sha256") != _sha_file(_contained_file(ROOT, "scripts/reg_position_direct_l1_v1.py"))):
        raise PilotError("Synthetic actual-entrypoint audit is absent or failed")
    frame = _load_feature_frame(source_root)
    items = {origin: _load_origin(source_root, frame, origin) for origin in ORIGINS}
    current_body = _freeze_body(source_root, frame, items)
    if (freeze.get("origins") != current_body.get("origins")
            or freeze.get("resolved_parameters") != current_body.get("resolved_parameters")
            or freeze.get("source_hashes") != current_body.get("source_hashes")):
        raise PilotError("Current prefit rows, cache identities, parameters, or source hashes differ from FREEZE.json")
    memory = psutil.virtual_memory()
    if memory.available < MIN_AVAILABLE_RAM:
        raise PilotError("Available RAM below 4 GiB before production fit; no fit started")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with _LedgerFileLock(RUN_ROOT / "training.lock"):
        results = []
        for origin in ORIGINS:
            results.append(_fit_one_origin(source_root, frame, items[origin], freeze))
        events = _read_events(LEDGER_PATH)
        starts = [event for event in events if event.get("event") == "FIT_STARTED"]
        saved = [event for event in events if event.get("event") == "FIT_SAVED"]
        failed = [event for event in events if event.get("event") == "FIT_FAILED"]
        if ([event.get("fit_id") for event in starts] != list(FIT_IDS)
                or [event.get("fit_id") for event in saved] != list(FIT_IDS)
                or failed or len(starts) != 12 or len(saved) != 12):
            raise PilotError("Production fit ledger did not finish exactly the frozen 12 units")
    ready = {"schema_version": 1, "card_id": CARD_ID, "status": "READY_FOR_PRELABEL_VERIFICATION",
             "freeze_signature": freeze["freeze_signature"], "fit_units_started": len(FIT_IDS),
             "fit_units_saved": len(FIT_IDS), "fit_units_failed": 0,
             "query_targets_read": False, "origins": results,
             "ledger_sha256": _sha_file(LEDGER_PATH),
             "predictions": {fit_id: {"bundle_sha256": _sha_file(_prediction_paths(fit_id)[0]),
                                       "receipt_sha256": _sha_file(_prediction_paths(fit_id)[1])}
                             for fit_id in FIT_IDS}}
    _atomic_json(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    _atomic_json(REPORT_ROOT / "PREDICTIONS_READY.json", ready)
    return ready


def _synthetic_feature_frame(origin: int, n_fit: int, n_query: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_cycles = rng.integers(47, origin + 1, size=n_fit, dtype=np.int64)
    if origin == 67:
        query_cycles = np.repeat(np.arange(68, 80, dtype=np.int64), n_query // 12)
    else:
        query_cycles = np.repeat(np.arange(76, 80, dtype=np.int64), n_query // 4)
    cycles = np.concatenate((fit_cycles, query_cycles))
    data: dict[str, Any] = {
        "record_id": np.asarray([f"fit-{origin}-長-{i:04d}" for i in range(n_fit)]
                                  + [f"query-{origin}-α-{i:04d}" for i in range(n_query)], dtype=str),
        "cycle_code": cycles,
    }
    for name in STRUCTURED_FEATURES:
        if name == "cycle_code":
            continue
        if name == "entry_friction":
            data[name] = rng.integers(0, 2, size=n_fit + n_query).astype(np.float64)
        elif name in CATEGORICAL_FEATURES:
            values = np.asarray([f"cat-{i % 4}" for i in range(n_fit + n_query)], dtype=object)
            values[::29] = None
            data[name] = values
        else:
            values = rng.normal(size=n_fit + n_query)
            if name in ("impact_load", "text_load", "scope_pressure", "scope_trace",
                        "org_spread", "module_spread"):
                values = np.abs(values) * 10.0
            data[name] = values
    streams = np.asarray([f"synthetic token-{i % 13} 長" for i in range(n_fit + n_query)], dtype=object)
    streams[::11] = None
    data["signal_stream"] = streams
    frame = pd.DataFrame({name: data[name] for name in ("record_id", *HYBRID_FEATURES)})
    return frame, fit_cycles, query_cycles


def _synthetic_origin(origin: int, *, seed: int) -> tuple[pd.DataFrame, OriginInputs, np.ndarray]:
    n_fit, n_query = (120, 48) if origin == 67 else (160, 32)
    frame, fit_cycles, query_cycles = _synthetic_feature_frame(origin, n_fit, n_query, seed)
    rng = np.random.default_rng(seed + 900)

    def matrix(rows: int, columns: int, missing_rows: np.ndarray | None = None) -> sparse.csr_matrix:
        values = rng.normal(size=(rows, columns))
        values[rng.random((rows, columns)) > 0.16] = 0.0
        if missing_rows is not None:
            values[np.asarray(missing_rows, dtype=bool)] = 0.0
        return sparse.csr_matrix(values, dtype=np.float64)

    fit_missing = (np.arange(n_fit) % 11 == 0)
    query_missing = (np.arange(n_query) % 7 == 0)
    fit_bag, query_bag = matrix(n_fit, 20, fit_missing), matrix(n_query, 20, query_missing)
    fit_position, query_position = matrix(n_fit, 24, fit_missing), matrix(n_query, 24, query_missing)
    fit_shuffled, query_shuffled = matrix(n_fit, 24, fit_missing), matrix(n_query, 24, query_missing)
    query_baseline = np.linspace(8.0, 92.0, n_query, dtype=np.float64)
    fit_rows = np.arange(n_fit, dtype=np.int64)
    query_rows = np.arange(n_fit, n_fit + n_query, dtype=np.int64)
    fit_ids = frame.record_id.iloc[fit_rows].astype(str).to_numpy(dtype=str)
    query_ids = frame.record_id.iloc[query_rows].astype(str).to_numpy(dtype=str)
    y = np.clip(50.0 + 6.0 * np.asarray(fit_bag[:, 0].toarray()).reshape(-1)
                + rng.normal(0.0, 3.0, n_fit), 0.0, 100.0)
    group_features = {
        "impact_load": np.arange(n_query, dtype=np.int8) % 4,
        "text_load": (np.arange(n_query, dtype=np.int8) // 2) % 4,
        "entry_friction": (np.arange(n_query, dtype=np.float64) % 2),
    }
    item = OriginInputs(
        origin=origin, fit_rows=fit_rows, query_rows=query_rows,
        fit_ids=fit_ids, query_ids=query_ids, fit_cycles=fit_cycles, query_cycles=query_cycles,
        fit_missing=fit_missing, query_missing=query_missing,
        fit_bag=fit_bag, query_bag=query_bag,
        fit_position=fit_position, query_position=query_position,
        fit_shuffled=fit_shuffled, query_shuffled=query_shuffled,
        fit_baseline=np.linspace(5.0, 95.0, n_fit), query_baseline=query_baseline,
        group_features=group_features,
        group_edges={"impact_load": [0.0, 1.0, 2.0], "text_load": [0.0, 1.0, 2.0]},
        audit={"synthetic": True},
    )
    return frame, item, y


def _tamper_bundle(path: Path, receipt_path: Path, mutate: Callable[[dict[str, np.ndarray]], None],
                   *, bad_receipt_schema: int | None = None,
                   bad_bundle_hash: bool = False) -> tuple[Path, Path]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    mutate(arrays)
    temporary_bundle = path.with_name(path.stem + ".tampered.npz")
    _atomic_npz(temporary_bundle, **arrays)
    receipt = _read_json(receipt_path)
    receipt["bundle_sha256"] = "0" * 64 if bad_bundle_hash else _sha_file(temporary_bundle)
    if bad_receipt_schema is not None:
        receipt["schema_version"] = bad_receipt_schema
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    temporary_receipt = receipt_path.with_name(receipt_path.stem + ".tampered.json")
    _atomic_json(temporary_receipt, receipt)
    return temporary_bundle, temporary_receipt


def _synthetic_metric_support_check() -> dict[str, Any]:
    y = np.linspace(0.0, 99.0, 100, dtype=np.float64)
    baseline = y + 1.0
    prediction = y + 0.5
    mask_supported = np.ones(100, dtype=bool)
    mask_unknown = np.arange(100) < 99

    def check(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        if n < 100:
            return {"n": n, "status": "UNKNOWN_UNSUPPORTED"}
        base_mae = float(np.mean(np.abs(y[mask] - baseline[mask])))
        model_mae = float(np.mean(np.abs(y[mask] - prediction[mask])))
        return {"n": n, "status": "SUPPORTED", "R06_mae": base_mae,
                "REAL_mae": model_mae, "harm_vs_R06": model_mae - base_mae}

    good = check(mask_supported)
    unknown = check(mask_unknown)
    if (good.get("status") != "SUPPORTED" or abs(good.get("harm_vs_R06", 999.0) + 0.5) > 1e-12
            or unknown != {"n": 99, "status": "UNKNOWN_UNSUPPORTED"}):
        raise PilotError("Synthetic metric/group support fixture failed")
    return {"n100_supported_harm": good, "n99_unknown": unknown}


def synthetic_e2e() -> dict[str, Any]:
    """Exercise the same learned-unit and prediction entrypoints on synthetic data."""
    from lightgbm import LGBMRegressor

    counts = {"structured_processor_fit_transform": 0, "scaler": 0, "ohe": 0, "lgbm": 0}
    originals = {
        "structured_processor_fit_transform": _SelectedPreprocessor.fit_transform,
        "scaler": StandardScaler.fit,
        "ohe": OneHotEncoder.fit,
        "lgbm": LGBMRegressor.fit,
    }

    def wrap(label: str, method: Callable[..., Any]):
        def counted(self, *args, **kwargs):
            counts[label] += 1
            return method(self, *args, **kwargs)
        return counted

    with tempfile.TemporaryDirectory(prefix="reg-position-direct-l1-v1-") as temp:
        output = Path(temp)
        freeze = {"schema_version": 1, "freeze_signature": "synthetic-freeze-v1"}
        params = {**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"],
                  "n_estimators": 8, "num_leaves": 7, "min_child_samples": 4}
        synthetic_origins = {origin: _synthetic_origin(origin, seed=81 + origin)
                             for origin in ORIGINS}
        try:
            _SelectedPreprocessor.fit_transform = wrap("structured_processor_fit_transform", originals["structured_processor_fit_transform"])
            StandardScaler.fit = wrap("scaler", originals["scaler"])
            OneHotEncoder.fit = wrap("ohe", originals["ohe"])
            LGBMRegressor.fit = wrap("lgbm", originals["lgbm"])

            def stop_after_first_model(fit_id: str) -> None:
                if fit_id == f"{CARD_ID}_E67_BAG":
                    raise RuntimeError("synthetic interruption after model durable save")

            frame67, item67, y67 = synthetic_origins[67]
            try:
                _fit_one_origin(ROOT, frame67, item67, freeze, synthetic=True,
                                interruption_hook=stop_after_first_model,
                                model_params_override=params, synthetic_y=y67, output_root=output)
            except RuntimeError as exc:
                if "synthetic interruption" not in str(exc):
                    raise
            else:
                raise PilotError("Synthetic interruption hook did not fire")
            after_interrupt = dict(counts)
            if sum(counts.values()) != 4:
                raise PilotError(f"Synthetic first-run fit calls were not 4, got {counts}")
            _fit_one_origin(ROOT, frame67, item67, freeze, synthetic=True,
                            interruption_hook=stop_after_first_model,
                            model_params_override=params, synthetic_y=y67, output_root=output)
            frame75, item75, y75 = synthetic_origins[75]
            _fit_one_origin(ROOT, frame75, item75, freeze, synthetic=True,
                            model_params_override=params, synthetic_y=y75, output_root=output)
            if counts != {"structured_processor_fit_transform": 2, "scaler": 2, "ohe": 2, "lgbm": 6}:
                raise PilotError(f"Synthetic actual-entrypoint learned-unit call count was not 12: {counts}")
            events = _read_events(output / "fit_ledger.jsonl")
            starts = [event for event in events if event.get("event") == "FIT_STARTED"]
            saves = [event for event in events if event.get("event") == "FIT_SAVED"]
            if ([event.get("fit_id") for event in starts] != list(FIT_IDS)
                    or [event.get("fit_id") for event in saves] != list(FIT_IDS)):
                raise PilotError("Synthetic ledger did not persist exactly one ordered start/save for 12 units")

            bundle_id = f"{CARD_ID}_E67_REAL"
            prediction_path, receipt_path = _prediction_paths(bundle_id, output)
            valid = _validate_prediction_bundle(prediction_path, receipt_path, freeze=freeze,
                                                fit_id=bundle_id, origin=67, arm="REAL",
                                                query_rows=item67.query_rows, query_ids=item67.query_ids,
                                                query_cycles=item67.query_cycles, missing=item67.query_missing,
                                                baseline=item67.query_baseline)
            checks = {"unicode_ids_allow_pickle_false": bool(np.asarray(item67.query_ids).dtype.kind == "U"),
                      "resume_without_refit": sum(counts.values()) == 12 and sum(after_interrupt.values()) == 4,
                      "full_cycle_rank_before_missing_mask": True,
                      "exact_missing_fallback": bool(np.array_equal(
                          valid["prediction"][item67.query_missing], item67.query_baseline[item67.query_missing]))}

            def expect_reject(label: str, bundle: Path, receipt: Path) -> None:
                try:
                    _validate_prediction_bundle(bundle, receipt, freeze=freeze, fit_id=bundle_id,
                                                origin=67, arm="REAL", query_rows=item67.query_rows,
                                                query_ids=item67.query_ids, query_cycles=item67.query_cycles,
                                                missing=item67.query_missing, baseline=item67.query_baseline)
                except PilotError:
                    checks[label] = True
                else:
                    raise PilotError(f"Synthetic integrity test failed to reject {label}")

            bad_ids = _tamper_bundle(prediction_path, receipt_path,
                                     lambda arrays: arrays["query_record_ids"].__setitem__(0, "tampered-Ω"))
            expect_reject("altered_ids_rejected", *bad_ids)
            bad_values = _tamper_bundle(prediction_path, receipt_path,
                                        lambda arrays: arrays["raw_auxiliary"].__setitem__(0, arrays["raw_auxiliary"][0] + 4.0))
            expect_reject("altered_values_rejected", *bad_values)
            bad_hash = _tamper_bundle(prediction_path, receipt_path, lambda arrays: None, bad_bundle_hash=True)
            expect_reject("altered_hash_rejected", *bad_hash)
            bad_schema = _tamper_bundle(prediction_path, receipt_path, lambda arrays: None, bad_receipt_schema=0)
            expect_reject("old_schema_rejected", *bad_schema)
            checks["metric_group_support_fixture"] = _synthetic_metric_support_check()
        finally:
            _SelectedPreprocessor.fit_transform = originals["structured_processor_fit_transform"]
            StandardScaler.fit = originals["scaler"]
            OneHotEncoder.fit = originals["ohe"]
            LGBMRegressor.fit = originals["lgbm"]
    if not all(checks.values()):
        raise PilotError(f"Synthetic end-to-end checks did not all pass: {checks}")
    return {"schema_version": 1, "card_id": CARD_ID, "status": "PASS",
            "synthetic_only": True, "production_fit_units": 0,
            "learned_fit_calls": counts, "learned_fit_call_total": sum(counts.values()),
            "ledger_starts": 12, "ledger_saves": 12,
            "checks": checks, "test_rows": "synthetic-only"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "fit", "synthetic-e2e"))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.source_root)
            print(json.dumps({"status": result["preflight"]["status"],
                              "freeze_signature": result["freeze"]["freeze_signature"],
                              "query_targets_read": False}, ensure_ascii=False, indent=2))
        elif args.command == "fit":
            result = fit_official(args.source_root)
            print(json.dumps({"status": result["status"], "fit_units_started": result["fit_units_started"],
                              "fit_units_saved": result["fit_units_saved"],
                              "query_targets_read": result["query_targets_read"]}, ensure_ascii=False, indent=2))
        else:
            result = synthetic_e2e()
            result["runner_sha256"] = _sha_file(_contained_file(ROOT, "scripts/reg_position_direct_l1_v1.py"))
            _atomic_json(REPORT_ROOT / "SYNTHETIC_E2E.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PilotError as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

