"""Bounded full-Train E79 exploratory inference for two REG candidates.

This runner does not read Test feature values until a trained-state freeze has
been independently verified. Test targets are never selected or loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
DATA_ROOT = SOURCE_ROOT / "data/raw/regression"
TRAIN_PATH = DATA_ROOT / "Train.csv"
TEST_PATH = DATA_ROOT / "Test.csv"
SAMPLE_PATH = DATA_ROOT / "SampleSubmission.csv"
ANCHOR_CSV = SOURCE_ROOT / "submissions/optimization_v3_expanded_v1/r06_original.csv"
ANCHOR_MANIFEST = SOURCE_ROOT / "reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json"
ANCHOR_RUN = SOURCE_ROOT / "experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL"
ANCHOR_PREDICTIONS = ANCHOR_RUN / "predictions.npz"
ANCHOR_FIT_STATE = ANCHOR_RUN / "fit_state.json"
PUBLIC_REGISTRY = SOURCE_ROOT / "reports/submissions/reconciliation_20261102/REGISTRY.json"
PROTOCOL_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PROTOCOL.json"
INITIAL_FREEZE_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PREFIT_TRAIN_FREEZE.json"
RESOURCE_CHECK_V2_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PREFIT_RESOURCE_CHECK_V2.json"
SUCCESSOR_FREEZE_V2_PATH = ROOT / "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PREFIT_TRAIN_FREEZE_V2.json"
FREEZE_PATH_ENV = "REG_CURRENT_R06_E79_PREFIT_FREEZE_PATH"
FREEZE_PATH = Path(os.environ.get(FREEZE_PATH_ENV, str(INITIAL_FREEZE_PATH))).resolve()
TRAINED_REPORT_DIR = ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
RUN_ROOT = ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
CSV_ROOT = ROOT / "submissions/reg_current_r06_e79_exploratory_v1"
TRAINED_REPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = (
    "experiments/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PROTOCOL.json",
    "scripts/optimization_v3_reg_current_r06_e79_exploratory_v1.py",
    "scripts/verify_optimization_v3_reg_current_r06_e79_exploratory_v1.py",
    "src/models/optimization_v3_reg_r12_current_route_late_v1.py",
    "src/models/optimization_v3_batch1_r12.py",
    "src/models/optimization_v2_regression.py",
    "src/models/classification_models.py",
    "src/models/positional_text_features.py",
    "src/models/regression_research_text.py",
    "src/models/regression_research_tabular.py",
    "src/models/regression_round2_models.py",
    "src/models/round1_models.py",
    "src/models/reg_position_l1_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "scripts/reg_position_direct_l1_v1.py",
    "reports/optimization_v3/REG_R12_CURRENT_ROUTE_LATE_V1/RESULTS.json",
    "reports/optimization_v3/reg_position_direct_l1_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_pilot_v1/RECIPE.json",
)

INPUT_FILES = {
    "official_regression_train": TRAIN_PATH,
    "official_regression_test": TEST_PATH,
    "official_regression_sample_submission": SAMPLE_PATH,
    "current_r06_e79_final_csv": ANCHOR_CSV,
    "current_r06_e79_manifest": ANCHOR_MANIFEST,
    "current_r06_e79_prediction_archive": ANCHOR_PREDICTIONS,
    "current_r06_e79_fit_state": ANCHOR_FIT_STATE,
    "r06_public_receipt_registry": PUBLIC_REGISTRY,
}


class IntegrityError(RuntimeError):
    """Fail-closed experiment integrity or accounting violation."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind in "OUS":
        data = canonical_bytes(array.astype(str).tolist())
    else:
        data = np.ascontiguousarray(array).tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def sha_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256(canonical_bytes({"shape": list(matrix.shape), "dtype": str(matrix.dtype)}))
    digest.update(np.asarray(matrix.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(matrix.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return to_builtin(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_builtin(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(to_builtin(event), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def git_text(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(["git", "-C", str(cwd), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def git_object_is_commit(root: Path, revision: str) -> bool:
    result = subprocess.run(["git", "-C", str(root), "cat-file", "-t", revision],
                            capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == "commit"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IntegrityError(f"Invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected JSON object: {path}")
    return value


def read_signed_json(path: Path) -> dict[str, Any]:
    value = read_json(path)
    body = {key: item for key, item in value.items()
            if key not in {"signature", "signature_method"}}
    if (value.get("signature_method") != "SHA-256 canonical JSON"
            or value.get("signature") != canonical_sha(body)):
        raise IntegrityError(f"Signed receipt signature mismatch: {path}")
    return value


def protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL_PATH)
    if value.get("namespace") != "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1":
        raise IntegrityError("Protocol namespace mismatch")
    return value


def current_source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise IntegrityError(f"Frozen source is missing: {relative}")
        hashes[relative] = sha_file(path)
    return hashes


def current_input_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in INPUT_FILES.items():
        if not path.is_file():
            raise IntegrityError(f"Required immutable input is missing: {path}")
        result[name] = sha_file(path)
    return result


def verify_anchor() -> dict[str, Any]:
    spec = protocol()["anchor"]
    manifest = read_json(ANCHOR_MANIFEST)
    if manifest.get("candidate") != spec["candidate"]:
        raise IntegrityError("R06 anchor candidate label differs")
    expected = {
        "manifest_sha256": sha_file(ANCHOR_MANIFEST),
        "csv_sha256": sha_file(ANCHOR_CSV),
        "prediction_archive_sha256": sha_file(ANCHOR_PREDICTIONS),
        "fit_state_sha256": sha_file(ANCHOR_FIT_STATE),
    }
    if expected != {key: spec[key] for key in expected}:
        raise IntegrityError(f"R06 anchor artifact hashes differ: {expected}")
    official = manifest.get("official_input_hashes", {})
    if (official.get("train") != spec["train_sha256"]
            or official.get("test") != spec["test_sha256"]
            or official.get("sample") != spec["sample_sha256"]):
        raise IntegrityError("R06 manifest does not bind the frozen official data")
    fit = manifest.get("fit_audit", {})
    if (fit.get("fit_rows") != 50608 or fit.get("fit_cycle_max") != 79
            or fit.get("train_input", {}).get("sha256") != spec["train_sha256"]
            or fit.get("test_labels_passed_to_fit") is not False):
        raise IntegrityError("R06 fitted state is not the all-Train E79 anchor")
    prediction_meta = manifest.get("prediction_audit", {})
    if (prediction_meta.get("prediction_sha256") != spec["final_prediction_digest"]
            or "complete Test cycle" not in prediction_meta.get("mapping", "")):
        raise IntegrityError("R06 manifest does not identify the final mapped predictions")
    if not git_object_is_commit(SOURCE_ROOT, spec["source_commit"]):
        raise IntegrityError("R06 producer source commit is absent from the read-only source repository")
    if not git_object_is_commit(SOURCE_ROOT, spec["execution_head"]):
        raise IntegrityError("R06 execution head is absent from the read-only source repository")

    registry = read_json(PUBLIC_REGISTRY)
    entries = registry.get("submissions", registry.get("entries", []))
    matches = [row for row in entries if row.get("candidate") == "R06_ORIGINAL"]
    if len(matches) != 1:
        raise IntegrityError("Historical public registry does not uniquely identify R06_ORIGINAL")
    public = matches[0]
    if (float(public.get("public_score")) != float(spec["public_mae_as_reported"])
            or public.get("local_csv_sha256") != spec["csv_sha256"]):
        raise IntegrityError("R06 public receipt and local final prediction CSV do not match")

    with np.load(ANCHOR_PREDICTIONS, allow_pickle=False) as archive:
        if "prediction" not in archive.files or "raw_candidate" not in archive.files:
            raise IntegrityError("R06 prediction archive lacks expected members")
        final = np.asarray(archive["prediction"], dtype=np.float64)
        raw = np.asarray(archive["raw_candidate"], dtype=np.float64)
        if final.shape != (29421,) or raw.shape != final.shape:
            raise IntegrityError("R06 prediction archive row count differs")
        if not np.isfinite(final).all() or np.any((final < 0) | (final > 100)):
            raise IntegrityError("R06 final mapped prediction values are invalid")
        if sha_array(final, dtype="<f8") != spec["final_prediction_digest"]:
            raise IntegrityError("R06 final mapped prediction digest mismatch")
    base = pd.read_csv(ANCHOR_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    if list(base.columns) != ["record_id", "threat_rank"] or len(base) != 29421:
        raise IntegrityError("R06 final CSV schema/row count mismatch")
    values = pd.to_numeric(base["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise IntegrityError("R06 final CSV values are invalid")
    if not np.array_equal(values, final):
        raise IntegrityError("R06 final CSV is not bit-exact to its final mapped prediction archive")
    return {
        "candidate": "R06_ORIGINAL",
        "public_mae_as_reported": spec["public_mae_as_reported"],
        "manifest_sha256": expected["manifest_sha256"],
        "csv_sha256": expected["csv_sha256"],
        "prediction_archive_sha256": expected["prediction_archive_sha256"],
        "fit_state_sha256": expected["fit_state_sha256"],
        "final_prediction_member": "prediction",
        "raw_candidate_member_used": False,
        "final_prediction_digest": spec["final_prediction_digest"],
        "csv_archive_max_abs_delta": float(np.max(np.abs(values - final))),
        "fit_rows": int(fit["fit_rows"]),
        "fit_cycles": [int(fit["fit_cycle_min"]), int(fit["fit_cycle_max"])],
        "source_commit": spec["source_commit"],
        "execution_head": spec["execution_head"],
        "status": "PASS_EXACT_PUBLIC_R06_E79_ANCHOR",
    }


def feature_only_train() -> pd.DataFrame:
    from src.models.regression_research_text import HYBRID_FEATURES
    columns = list(dict.fromkeys(("record_id", *HYBRID_FEATURES)))
    frame = pd.read_csv(TRAIN_PATH, usecols=columns, dtype={"record_id": "string"})
    if len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise IntegrityError("Train feature-only population/ID check failed")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise IntegrityError("Train feature-only cycles are not exactly 47-79")
    return frame


def memory_snapshot() -> dict[str, float]:
    memory = psutil.virtual_memory()
    return {"available_gib": float(memory.available / (1024**3)),
            "total_gib": float(memory.total / (1024**3))}


def resource_estimates(frame: pd.DataFrame) -> dict[str, Any]:
    from src.models.regression_research_text import _text_documents, _TFIDF_PARAMS as A_PARAMS
    from src.models.optimization_v3_positional_text_pilot_v1 import _TFIDF_PARAMS as B_PARAMS
    docs, _ = _text_documents(frame["signal_stream"])
    a_analyzer = TfidfVectorizer(**A_PARAMS).build_analyzer()
    b_analyzer = TfidfVectorizer(**B_PARAMS).build_analyzer()
    a_occurrences = int(sum(len(a_analyzer(doc)) for doc in docs))
    b_ngrams = 0
    b_unigrams = 0
    for doc in docs:
        tokens = b_analyzer(doc)
        b_ngrams += len(tokens)
        b_unigrams += sum(" " not in token for token in tokens)
    rows = len(frame)
    # The structured upper bound reserves 64 nonzero values per row, plus the
    # explicit missing-stream indicator. The text bounds count every analyzer
    # occurrence before TF-IDF pruning and duplicate coalescing.
    structured_nnz = rows * 65
    bytes_per_nnz = 16
    row_pointers = (rows + 1) * 8
    a_csr_gib = ((a_occurrences + structured_nnz) * bytes_per_nnz + row_pointers) / (1024**3)
    b_csr_gib = ((b_ngrams + b_unigrams + structured_nnz) * bytes_per_nnz + row_pointers) / (1024**3)
    return {
        "fit_rows": int(rows),
        "fit_cycles": [47, 79],
        "A_word_ngram_occurrence_upper_bound": a_occurrences,
        "B_bag_ngram_occurrence_upper_bound": int(b_ngrams),
        "B_position_unigram_occurrence_upper_bound": int(b_unigrams),
        "structured_nnz_upper_bound": int(structured_nnz),
        "estimate_method": "2.25GiB fixed/LightGBM allowance plus four copies of CSR design upper bound; word counts include every analyzer occurrence before pruning; structured bound is 65 nonzeros per row.",
        "A_sparse_design_upper_gib": float(a_csr_gib),
        "B_sparse_design_upper_gib": float(b_csr_gib),
        "A_estimated_peak_gib": float(2.25 + 4.0 * a_csr_gib),
        "B_estimated_peak_gib": float(2.25 + 4.0 * b_csr_gib),
        "max_job_rss_gib": 6.0,
        "minimum_available_ram_gib": 4.0,
        "minimum_projected_reserve_gib": 2.0,
        "jobs": 1,
        "threads_per_fit": 2,
        "predictor_fits_performed": 0,
    }


def prefit_freeze() -> dict[str, Any]:
    if INITIAL_FREEZE_PATH.exists():
        raise IntegrityError("Prefit freeze already exists; refusing overwrite or retry")
    if git_text("status", "--porcelain"):
        raise IntegrityError("Source/protocol worktree must be clean before prefit freeze")
    head = git_text("rev-parse", "HEAD")
    if not git_object_is_commit(ROOT, head):
        raise IntegrityError("REG source commit cannot be resolved")
    frame = feature_only_train()
    anchor = verify_anchor()
    resources = resource_estimates(frame)
    memory = memory_snapshot()
    largest = max(resources["A_estimated_peak_gib"], resources["B_estimated_peak_gib"])
    resource_pass = (
        memory["available_gib"] >= 4.0
        and largest <= min(6.0, memory["available_gib"] - 2.0)
    )
    body = {
        "schema_version": 1,
        "namespace": "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1",
        "status": "PREFIT_TRAIN_FREEZE_PASS" if resource_pass else "BLOCKED_RESOURCE",
        "source_commit": head,
        "source_sha256": current_source_hashes(),
        "input_sha256": current_input_hashes(),
        "anchor_verification": anchor,
        "train_feature_only": {
            "path": str(TRAIN_PATH),
            "sha256": sha_file(TRAIN_PATH),
            "rows": int(len(frame)),
            "ordered_record_ids_sha256": sha_array(frame["record_id"].astype(str).to_numpy(dtype=str)),
            "ordered_cycle_sha256": sha_array(pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype="<i8")),
            "target_values_loaded": False,
            "test_features_loaded": False,
            "test_labels_loaded": False,
        },
        "learned_units_plan": [
            {"candidate": "A", "id": "A_TFIDF", "kind": "preprocessing"},
            {"candidate": "A", "id": "A_STRUCTURED", "kind": "preprocessing"},
            {"candidate": "A", "id": "A_SCALER", "kind": "preprocessing"},
            {"candidate": "A", "id": "A_OHE", "kind": "preprocessing"},
            {"candidate": "A", "id": "A_R12_RANKER", "kind": "predictor"},
            {"candidate": "B", "id": "B_SHARED_WORD_TFIDF", "kind": "preprocessing"},
            {"candidate": "B", "id": "B_STRUCTURED", "kind": "preprocessing"},
            {"candidate": "B", "id": "B_SCALER", "kind": "preprocessing"},
            {"candidate": "B", "id": "B_OHE", "kind": "preprocessing"},
            {"candidate": "B", "id": "B_DIRECT_POSITION_REAL_REGRESSOR", "kind": "predictor"},
        ],
        "learned_units_max": 10,
        "predictor_fits_max": 2,
        "preprocessing_fits_max": 8,
        "preprocessing_fits_planned": 8,
        "predictor_fits_planned": 2,
        "resource_preflight": {**resources, "snapshot": memory, "status": "PASS" if resource_pass else "FAIL"},
        "boundaries": {
            "official_test_ids_loaded": False,
            "official_test_features_loaded": False,
            "test_labels_loaded": False,
            "kaggle_accessed": False,
            "r06_refit": False,
            "candidate_outcomes_used_to_select_recipe": False,
        },
    }
    signed = {**body, "signature": canonical_sha(body), "signature_method": "SHA-256 canonical JSON"}
    write_json_once(INITIAL_FREEZE_PATH, signed)
    return signed


def successor_prefit_freeze_v2() -> dict[str, Any]:
    """Bind one newly captured resource check to a versioned successor freeze."""
    if SUCCESSOR_FREEZE_V2_PATH.exists():
        raise IntegrityError("Versioned successor freeze already exists; overwrite forbidden")
    if git_text("status", "--porcelain"):
        raise IntegrityError("Source and resource-attestation commit must be clean before successor freeze")
    initial = read_signed_json(INITIAL_FREEZE_PATH)
    if initial.get("status") != "BLOCKED_RESOURCE":
        raise IntegrityError("Successor freeze requires the preserved initial BLOCKED_RESOURCE receipt")
    attestation = read_signed_json(RESOURCE_CHECK_V2_PATH)
    if attestation.get("status") != "PREFIT_RESOURCE_CHECK_PASS":
        raise IntegrityError("Fresh versioned resource attestation did not pass")
    if (attestation.get("supersedes_blocked_freeze_signature") != initial.get("signature")
            or attestation.get("initial_blocked_freeze_sha256") != sha_file(INITIAL_FREEZE_PATH)
            or attestation.get("source_commit") != initial.get("source_commit")
            or attestation.get("source_sha256") != initial.get("source_sha256")
            or attestation.get("input_sha256") != initial.get("input_sha256")):
        raise IntegrityError("Resource attestation is not bound to the initial frozen lineage")
    if attestation.get("fit_ledger_exists_before_gate") is not False:
        raise IntegrityError("Resource attestation reports an existing fit ledger")
    if (attestation.get("test_features_loaded") is not False
            or attestation.get("test_labels_loaded") is not False
            or attestation.get("predictor_fits_performed") != 0
            or attestation.get("preprocessing_fits_performed") != 0):
        raise IntegrityError("Resource attestation crossed the pre-fit boundary")

    current_sources = current_source_hashes()
    current_inputs = current_input_hashes()
    if current_inputs != initial["input_sha256"]:
        raise IntegrityError("Official input bytes differ from the immutable initial freeze")
    allowed_operational_updates = {
        "scripts/optimization_v3_reg_current_r06_e79_exploratory_v1.py",
        "scripts/verify_optimization_v3_reg_current_r06_e79_exploratory_v1.py",
    }
    source_differences = [path for path in current_sources
                          if current_sources[path] != initial["source_sha256"].get(path)]
    if set(source_differences) != allowed_operational_updates:
        raise IntegrityError(f"Unexpected recipe/source changes since initial freeze: {source_differences}")
    if attestation.get("available_gib") is None:
        raise IntegrityError("Resource attestation lacks its one fresh RAM snapshot")
    available = float(attestation["available_gib"])
    old_resources = initial["resource_preflight"]
    peaks = {
        "A": float(old_resources["A_estimated_peak_gib"]),
        "B": float(old_resources["B_estimated_peak_gib"]),
    }
    if attestation.get("estimated_peak_gib") != peaks:
        raise IntegrityError("Resource peaks differ from the immutable initial estimate")
    projected = {candidate: available - peak for candidate, peak in peaks.items()}
    limits = attestation.get("resource_limits", {})
    if (available < float(limits.get("minimum_available_gib", 4.0))
            or any(value < float(limits.get("minimum_projected_reserve_gib", 2.0))
                   for value in projected.values())
            or any(value > float(limits.get("maximum_job_rss_gib", 6.0))
                   for value in peaks.values())):
        raise IntegrityError("Fresh resource attestation no longer satisfies the fixed gate")
    if git_text("status", "--porcelain"):
        raise IntegrityError("Worktree became dirty while successor lineage was being checked")
    head = git_text("rev-parse", "HEAD")
    resource_preflight = {
        **old_resources,
        "snapshot": {"available_gib": available,
                     "total_gib": float(old_resources["snapshot"]["total_gib"])},
        "projected_reserve_gib": projected,
        "process_snapshot": attestation["process_snapshot"],
        "fresh_resource_attestation_signature": attestation["signature"],
        "fresh_resource_attestation_sha256": sha_file(RESOURCE_CHECK_V2_PATH),
        "status": "PASS",
    }
    body = {key: value for key, value in initial.items()
            if key not in {"signature", "signature_method"}}
    body.update({
        "version": 2,
        "status": "PREFIT_TRAIN_FREEZE_PASS",
        "source_commit": head,
        "source_sha256": current_sources,
        "input_sha256": current_inputs,
        "resource_preflight": resource_preflight,
        "initial_blocked_freeze": {
            "path": INITIAL_FREEZE_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha_file(INITIAL_FREEZE_PATH),
            "signature": initial["signature"],
            "status": initial["status"],
        },
        "resource_attestation": {
            "path": RESOURCE_CHECK_V2_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha_file(RESOURCE_CHECK_V2_PATH),
            "signature": attestation["signature"],
        },
        "source_changes_since_initial_blocked_freeze": source_differences,
    })
    signed = {**body, "signature": canonical_sha(body),
              "signature_method": "SHA-256 canonical JSON"}
    write_json_once(SUCCESSOR_FREEZE_V2_PATH, signed)
    return signed


def validate_prefit_freeze(*, rehash: bool = True) -> dict[str, Any]:
    freeze = read_signed_json(FREEZE_PATH)
    if freeze.get("status") != "PREFIT_TRAIN_FREEZE_PASS":
        raise IntegrityError("Prefit freeze did not pass; no fit may start")
    if freeze.get("predictor_fits_max") != 2 or freeze.get("preprocessing_fits_max") != 8:
        raise IntegrityError("Frozen learned-unit caps changed")
    if rehash:
        if freeze.get("version") == 2:
            attestation_meta = freeze.get("resource_attestation", {})
            attestation_path = ROOT / attestation_meta.get("path", "")
            attestation = read_signed_json(attestation_path)
            if (sha_file(attestation_path) != attestation_meta.get("sha256")
                    or attestation.get("signature") != attestation_meta.get("signature")
                    or attestation.get("status") != "PREFIT_RESOURCE_CHECK_PASS"):
                raise IntegrityError("Successor freeze resource attestation is missing or changed")
            if git_text("rev-parse", "HEAD") != freeze.get("source_commit"):
                raise IntegrityError("Successor freeze is not bound to the current source commit")
        if current_source_hashes() != freeze["source_sha256"]:
            raise IntegrityError("Frozen source hashes changed after prefit freeze")
        if current_input_hashes() != freeze["input_sha256"]:
            raise IntegrityError("Frozen input hashes changed after prefit freeze")
    return freeze


def _ledger_path() -> Path:
    return RUN_ROOT / "fit_ledger.jsonl"


def _read_ledger() -> list[dict[str, Any]]:
    path = _ledger_path()
    if not path.exists():
        return []
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"Fit ledger JSON error at line {line_no}") from exc
    return events


def _record_started(fit_id: str, candidate: str, kind: str, order: int) -> None:
    events = _read_ledger()
    if any(event.get("fit_id") == fit_id and event.get("event") == "FIT_STARTED" for event in events):
        raise IntegrityError(f"Fit id {fit_id} already started; retry forbidden")
    if sum(event.get("event") == "FIT_STARTED" for event in events) >= 10:
        raise IntegrityError("Global learned-unit cap 10 would be exceeded")
    append_jsonl(_ledger_path(), {
        "event": "FIT_STARTED", "fit_id": fit_id, "candidate": candidate,
        "kind": kind, "order": order, "attempt": 1,
        "source_commit": git_text("rev-parse", "HEAD"),
        "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "ram_available_gib": memory_snapshot()["available_gib"],
    })


def _record_saved(fit_id: str, candidate: str, kind: str, artifact: Path, details: dict[str, Any]) -> dict[str, Any]:
    digest = sha_file(artifact)
    receipt = {
        "fit_id": fit_id, "candidate": candidate, "kind": kind,
        "artifact_path": artifact.relative_to(ROOT).as_posix(),
        "artifact_sha256": digest, "details": to_builtin(details),
    }
    append_jsonl(_ledger_path(), {
        "event": "FIT_SAVED", **receipt,
        "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return receipt


def _save_joblib(fit_id: str, candidate: str, kind: str, value: Any,
                 details: dict[str, Any], order: int) -> dict[str, Any]:
    artifact = RUN_ROOT / "units" / f"{fit_id}.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise IntegrityError(f"Artifact already exists; overwrite forbidden: {fit_id}")
    _record_started(fit_id, candidate, kind, order)
    try:
        joblib.dump(value, artifact, compress=0)
        with artifact.open("rb+") as stream:
            os.fsync(stream.fileno())
        return _record_saved(fit_id, candidate, kind, artifact, details)
    except Exception as exc:
        append_jsonl(_ledger_path(), {
            "event": "FIT_FAILED", "fit_id": fit_id, "candidate": candidate,
            "kind": kind, "attempt": 1, "exception_type": type(exc).__name__,
            "retry_performed": False, "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        raise


def _persist_started_unit(fit_id: str, candidate: str, kind: str, value: Any,
                          details: dict[str, Any]) -> dict[str, Any]:
    artifact = RUN_ROOT / "units" / f"{fit_id}.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if artifact.exists():
        raise IntegrityError(f"Artifact already exists; overwrite forbidden: {fit_id}")
    joblib.dump(value, artifact, compress=0)
    with artifact.open("rb+") as stream:
        os.fsync(stream.fileno())
    return _record_saved(fit_id, candidate, kind, artifact, details)


def _mark_group_started(ids: list[tuple[str, str]], candidate: str, first_order: int) -> None:
    for offset, (fit_id, kind) in enumerate(ids):
        _record_started(fit_id, candidate, kind, first_order + offset)


def _group_saved(ids: list[tuple[str, str, Any, dict[str, Any]]], candidate: str,
                 first_order: int) -> list[dict[str, Any]]:
    receipts = []
    for offset, (fit_id, kind, value, details) in enumerate(ids):
        artifact = RUN_ROOT / "units" / f"{fit_id}.joblib"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            raise IntegrityError(f"Artifact already exists; overwrite forbidden: {fit_id}")
        joblib.dump(value, artifact, compress=0)
        with artifact.open("rb+") as stream:
            os.fsync(stream.fileno())
        receipt = _record_saved(fit_id, candidate, kind, artifact, details)
        receipt["order"] = first_order + offset
        receipts.append(receipt)
    return receipts


def _candidate_paths(candidate: str) -> tuple[Path, Path, Path]:
    code = candidate.upper()
    if code not in {"A", "B"}:
        raise IntegrityError("Candidate must be A or B")
    report = TRAINED_REPORT_DIR / f"TRAINED_STATE_FREEZE_{code}.json"
    prelabel = TRAINED_REPORT_DIR / f"PRELABEL_PASS_{code}.json"
    manifest = TRAINED_REPORT_DIR / f"INFERENCE_MANIFEST_{code}.json"
    return report, prelabel, manifest


def _load_training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    from src.models.regression_research_text import HYBRID_FEATURES
    cols = list(dict.fromkeys(("record_id", *HYBRID_FEATURES, "threat_rank")))
    frame = pd.read_csv(TRAIN_PATH, usecols=cols, dtype={"record_id": "string"})
    if len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise IntegrityError("Train fit population/ID check failed")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise IntegrityError("Train fit cycles differ from 47-79")
    y = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise IntegrityError("Official Train target must be finite and within [0,100]")
    return frame, y


def _fit_preprocessor_group(candidate: str, frame: pd.DataFrame, first_order: int):
    from src.models.regression_research_text import RegressionTextModel
    ids = [
        (f"{candidate}_STRUCTURED", "structured_processor"),
        (f"{candidate}_SCALER", "numeric_scaler"),
        (f"{candidate}_OHE", "categorical_ohe"),
    ]
    _mark_group_started(ids, candidate, first_order)
    try:
        structured, processor, scaler, encoder, numeric_positions, state = RegressionTextModel._structured_fit(frame)
    except Exception as exc:
        for fit_id, kind in ids:
            append_jsonl(_ledger_path(), {
                "event": "FIT_FAILED", "fit_id": fit_id, "candidate": candidate,
                "kind": kind, "attempt": 1, "exception_type": type(exc).__name__,
                "retry_performed": False, "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
        raise
    details = {
        "fit_rows": int(len(frame)),
        "feature_schema": list(frame.columns),
        "numeric_positions": numeric_positions,
        "processor_audit": state,
        "structured_matrix_shape": list(structured.shape),
        "structured_matrix_sha256": sha_csr(structured),
        "actual_learned_calls": [
            "RegressionTextModel._structured_fit -> _SelectedPreprocessor.fit_transform x1",
            "StandardScaler.fit_transform x1",
            "OneHotEncoder.fit_transform x1",
        ],
    }
    objects = [
        (ids[0][0], ids[0][1], processor, details),
        (ids[1][0], ids[1][1], scaler, details),
        (ids[2][0], ids[2][1], encoder, details),
    ]
    receipts = _group_saved(objects, candidate, first_order)
    return structured, processor, scaler, encoder, numeric_positions, state, receipts


def _candidate_output_prediction(candidate: str, raw: np.ndarray, cycles: np.ndarray,
                                 missing: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    if raw.shape != cycles.shape or missing.shape != cycles.shape or baseline.shape != cycles.shape:
        raise IntegrityError("Inference vectors are not aligned")
    if not np.isfinite(raw).all() or not np.isfinite(baseline).all():
        raise IntegrityError("Inference vectors contain nonfinite values")
    unique = np.unique(cycles)
    if unique.size == 0 or np.any((unique < 80) | (unique > 91)):
        raise IntegrityError("Inference cycles must lie within the official Test cycle range 80-91")
    percentile = np.empty(len(raw), dtype=np.float64)
    for cycle in unique:
        mask = cycles == cycle
        percentile[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    if candidate.upper() == "A":
        result = np.clip(0.75 * baseline + 0.25 * percentile, 0.0, 100.0)
    else:
        auxiliary_rank50 = 0.5 * raw + 0.5 * percentile
        result = np.clip(0.75 * baseline + 0.25 * auxiliary_rank50, 0.0, 100.0)
    result[missing] = baseline[missing]
    return result


def _unit_artifact(fit_id: str) -> Path:
    return RUN_ROOT / "units" / f"{fit_id}.joblib"


def _load_unit(fit_id: str) -> Any:
    path = _unit_artifact(fit_id)
    if not path.is_file():
        raise IntegrityError(f"Missing trained unit artifact {fit_id}")
    return joblib.load(path)


def fit_candidate(candidate: str) -> dict[str, Any]:
    candidate = candidate.upper()
    freeze = validate_prefit_freeze()
    if candidate == "B":
        prior = TRAINED_REPORT_DIR / "CSV_QA_A.json"
        qa = read_signed_json(prior)
        prior_manifest = read_signed_json(TRAINED_REPORT_DIR / "INFERENCE_MANIFEST_A.json")
        prior_csv = CSV_ROOT / "r12_current_route_e79.csv"
        if (qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
                or qa.get("candidate") != "A"
                or qa.get("inference_manifest_signature") != prior_manifest.get("signature")
                or qa.get("prefit_freeze_signature") != freeze.get("signature")
                or qa.get("input_sha256") != freeze.get("input_sha256")
                or qa.get("source_sha256") != freeze.get("source_sha256")
                or prior_manifest.get("prefit_freeze_signature") != freeze.get("signature")
                or prior_manifest.get("input_sha256") != freeze.get("input_sha256")
                or prior_manifest.get("source_sha256") != freeze.get("source_sha256")
                or not prior_csv.is_file()
                or qa.get("csv_sha256") != sha_file(prior_csv)):
            raise IntegrityError("Candidate B is blocked until candidate A CSV QA passes")
    expected_before = 0 if candidate == "A" else 5
    prior_starts = [event for event in _read_ledger() if event.get("event") == "FIT_STARTED"]
    if len(prior_starts) != expected_before:
        raise IntegrityError(f"Unexpected prior fit starts before {candidate}: {len(prior_starts)}")
    if any(event.get("event") == "FIT_FAILED" for event in _read_ledger()):
        raise IntegrityError("A prior failed unit blocks all later fits")

    resources = freeze["resource_preflight"]
    peak = float(resources["A_estimated_peak_gib"] if candidate == "A" else resources["B_estimated_peak_gib"])
    available = float(resources["snapshot"]["available_gib"])
    if (resources.get("status") != "PASS" or available < 4.0
            or peak > min(6.0, available - 2.0)):
        raise IntegrityError(
            f"Resource gate blocked candidate {candidate}: frozen_available={available:.3f}GiB "
            f"estimated_peak={peak:.3f}GiB reserve=2GiB"
        )

    frame, y = _load_training_frame()
    from src.models.regression_research_text import (
        HYBRID_FEATURES, RegressionTextModel, _text_documents,
    )
    from src.models.optimization_v3_reg_r12_current_route_late_v1 import (
        LABEL_GAIN, RANKER_BASE_PARAMETERS, _select_features, grouped_fit_order,
    )
    from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS
    from src.models.optimization_v3_positional_text_pilot_v1 import (
        _TFIDF_PARAMS as B_TFIDF_PARAMS, build_position_block,
    )
    from src.models.reg_position_l1_v1 import unigram_bag
    from scipy.stats import rankdata as _rankdata
    from lightgbm import LGBMRanker, LGBMRegressor

    start_order = expected_before + 1
    receipts: list[dict[str, Any]] = []
    docs, missing = _text_documents(frame["signal_stream"])

    if candidate == "A":
        from src.models.regression_research_text import _TFIDF_PARAMS as A_TFIDF_PARAMS
        _record_started("A_TFIDF", candidate, "preprocessing", start_order)
        try:
            vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(docs)
            if vectorizer is None or fallback is not None:
                raise IntegrityError("R12 E79 word TF-IDF state unexpectedly fell back")
            receipts.append(_persist_started_unit(
                "A_TFIDF", candidate, "preprocessing", vectorizer,
                {"fit_rows": len(frame), "parameters": A_TFIDF_PARAMS,
                 "vocabulary_size": len(vectorizer.vocabulary_), "idf_sha256": sha_array(vectorizer.idf_),
                 "fit_transform_shape": list(text_matrix.shape), "fit_transform_sha256": sha_csr(text_matrix)},
            ))
        except Exception as exc:
            append_jsonl(_ledger_path(), {
                "event": "FIT_FAILED", "fit_id": "A_TFIDF", "candidate": candidate,
                "kind": "preprocessing", "attempt": 1, "exception_type": type(exc).__name__,
                "retry_performed": False, "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            raise
        structured, processor, scaler, encoder, numeric_positions, state, preproc_receipts = (
            _fit_preprocessor_group(candidate, frame, start_order + 1)
        )
        receipts.extend(preproc_receipts)
        missing_col = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_col), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        order, inverse_order, group_sizes, grades = grouped_fit_order(
            pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64), y
        )
        model = LGBMRanker(**{
            **dict(RANKER_BASE_PARAMETERS),
            "label_gain": list(LABEL_GAIN),
            "lambdarank_truncation_level": int(np.max(group_sizes)),
        })
        _record_started("A_R12_RANKER", candidate, "predictor", start_order + 4)
        try:
            with threadpool_limits(limits=2):
                model.fit(design[order].tocsr(), grades, group=group_sizes.tolist())
        except Exception as exc:
            append_jsonl(_ledger_path(), {
                "event": "FIT_FAILED", "fit_id": "A_R12_RANKER", "candidate": candidate,
                "kind": "predictor", "attempt": 1, "exception_type": type(exc).__name__,
                "retry_performed": False, "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            raise
        model_details = {
            "fit_rows": len(frame), "target_sha256": sha_array(y, dtype="<f8"),
            "grade_sha256": sha_array(grades, dtype="<i4"),
            "fit_cycle_counts": {str(int(c)): int(n) for c, n in zip(np.unique(frame["cycle_code"]), np.unique(frame["cycle_code"], return_counts=True)[1])},
            "group_sizes_sha256": sha_array(group_sizes, dtype="<i4"),
            "design_shape": list(design.shape), "design_csr_sha256": sha_csr(design),
            "parameters": model.get_params(deep=False), "validation_or_early_stopping": False,
        }
        receipts.append(_save_fitted_model("A_R12_RANKER", candidate, model, model_details))
        model_id = "A_R12_RANKER"
        transform_state = {
            "vectorizer": vectorizer, "processor": processor, "scaler": scaler,
            "encoder": encoder, "numeric_positions": numeric_positions,
        }
    else:
        vectorizer = TfidfVectorizer(**B_TFIDF_PARAMS)
        _record_started("B_SHARED_WORD_TFIDF", candidate, "preprocessing", start_order)
        try:
            vectorizer.fit(docs)
            if not vectorizer.vocabulary_:
                raise IntegrityError("Candidate B full-Train word vectorizer has an empty vocabulary")
            receipts.append(_persist_started_unit(
                "B_SHARED_WORD_TFIDF", candidate, "preprocessing", vectorizer,
                {"fit_rows": len(frame), "parameters": B_TFIDF_PARAMS,
                 "vocabulary_size": len(vectorizer.vocabulary_), "idf_sha256": sha_array(vectorizer.idf_),
                 "shared_for": ["unigram_bag", "REAL_three_third_position"],
                 "position_transform_is_fit_free": True},
            ))
        except Exception as exc:
            append_jsonl(_ledger_path(), {
                "event": "FIT_FAILED", "fit_id": "B_SHARED_WORD_TFIDF", "candidate": candidate,
                "kind": "preprocessing", "attempt": 1, "exception_type": type(exc).__name__,
                "retry_performed": False, "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            raise
        structured, processor, scaler, encoder, numeric_positions, state, preproc_receipts = (
            _fit_preprocessor_group(candidate, frame, start_order + 1)
        )
        receipts.extend(preproc_receipts)
        bag = unigram_bag(vectorizer, docs)
        position, position_audit = build_position_block(docs, vectorizer, shuffled=False)
        missing_col = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, bag, position, missing_col), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        params = {**COMMON_PARAMETERS, **PARAMETER_SETS["regularized"]}
        model = LGBMRegressor(**params)
        _record_started("B_DIRECT_POSITION_REAL_REGRESSOR", candidate, "predictor", start_order + 4)
        try:
            with threadpool_limits(limits=2):
                model.fit(design, y)
        except Exception as exc:
            append_jsonl(_ledger_path(), {
                "event": "FIT_FAILED", "fit_id": "B_DIRECT_POSITION_REAL_REGRESSOR",
                "candidate": candidate, "kind": "predictor", "attempt": 1,
                "exception_type": type(exc).__name__, "retry_performed": False,
                "time_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            raise
        model_details = {
            "fit_rows": len(frame), "target": "direct threat_rank",
            "target_sha256": sha_array(y, dtype="<f8"),
            "design_shape": list(design.shape), "design_csr_sha256": sha_csr(design),
            "bag_csr_sha256": sha_csr(bag), "position_csr_sha256": sha_csr(position),
            "position_audit": position_audit, "parameters": model.get_params(deep=False),
            "validation_or_early_stopping": False,
        }
        receipts.append(_save_fitted_model("B_DIRECT_POSITION_REAL_REGRESSOR", candidate, model, model_details))
        model_id = "B_DIRECT_POSITION_REAL_REGRESSOR"
        transform_state = {
            "vectorizer": vectorizer, "processor": processor, "scaler": scaler,
            "encoder": encoder, "numeric_positions": numeric_positions,
        }

    events = _read_ledger()
    started = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    if len(started) != expected_before + 5 or len(saved) != len(started):
        raise IntegrityError("Candidate learned-unit lifecycle did not save its exact five starts")
    if len(started) > 10 or any(event.get("event") == "FIT_FAILED" for event in events):
        raise IntegrityError("Global cap/failure ledger gate failed")

    run_state = RUN_ROOT / f"state_{candidate}.joblib"
    if run_state.exists():
        raise IntegrityError("Trained-state bundle exists; overwrite forbidden")
    joblib.dump({"candidate": candidate, "model_id": model_id, "transform": transform_state}, run_state, compress=0)
    with run_state.open("rb+") as stream:
        os.fsync(stream.fileno())
    report_path, _, _ = _candidate_paths(candidate)
    trained_body = {
        "schema_version": 1,
        "namespace": "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1",
        "candidate": candidate,
        "candidate_id": protocol()["candidate_A" if candidate == "A" else "candidate_B"]["id"],
        "status": "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY",
        "prefit_freeze_signature": freeze["signature"],
        "source_commit": freeze["source_commit"],
        "current_fit_commit": git_text("rev-parse", "HEAD"),
        "input_sha256": freeze["input_sha256"],
        "source_sha256": freeze["source_sha256"],
        "fit_receipts": receipts,
        "candidate_state_path": run_state.relative_to(ROOT).as_posix(),
        "candidate_state_sha256": sha_file(run_state),
        "target_sha256": sha_array(y, dtype="<f8"),
        "train_rows": int(len(frame)),
        "train_record_ids_sha256": sha_array(frame["record_id"].astype(str).to_numpy(dtype=str)),
        "train_cycles": [47, 79],
        "predictor_fits_total": 1,
        "preprocessing_fits_candidate": 4,
        "global_predictor_fits_total": 1 if candidate == "A" else 2,
        "global_preprocessing_fits_total": 4 if candidate == "A" else 8,
        "global_learned_units_total": 5 if candidate == "A" else 10,
        "test_features_loaded": False,
        "test_labels_loaded": False,
        "kaggle_accessed": False,
        "retry_performed": False,
    }
    signed = {**trained_body, "signature": canonical_sha(trained_body),
              "signature_method": "SHA-256 canonical JSON"}
    write_json_once(report_path, signed)
    return signed


def _save_fitted_model(fit_id: str, candidate: str, model: Any,
                       details: dict[str, Any]) -> dict[str, Any]:
    artifact = _unit_artifact(fit_id)
    if artifact.exists():
        raise IntegrityError(f"Artifact already exists; overwrite forbidden: {fit_id}")
    joblib.dump(model, artifact, compress=0)
    with artifact.open("rb+") as stream:
        os.fsync(stream.fileno())
    return _record_saved(fit_id, candidate, "predictor", artifact, details)


def _text_missing_mask(series: pd.Series) -> np.ndarray:
    from src.models.classification_models import _text_values
    _, missing = _text_values(series)
    return np.asarray(missing, dtype=bool)


def _infer_design(candidate: str, test: pd.DataFrame, state: dict[str, Any]) -> tuple[Any, np.ndarray]:
    from src.models.regression_research_text import (
        RegressionTextModel, _text_documents,
    )
    docs, missing = _text_documents(test["signal_stream"])
    structured, _ = RegressionTextModel._structured_transform(
        test, state["processor"], state["scaler"], state["encoder"], state["numeric_positions"]
    )
    missing_col = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
    if candidate == "A":
        vectorizer = state["vectorizer"]
        text = vectorizer.transform(docs).tocsr().astype(np.float64)
        design = sparse.hstack((structured, text, missing_col), format="csr", dtype=np.float64)
    else:
        from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
        from src.models.reg_position_l1_v1 import unigram_bag
        vectorizer = state["vectorizer"]
        bag = unigram_bag(vectorizer, docs)
        position, _ = build_position_block(docs, vectorizer, shuffled=False)
        design = sparse.hstack((structured, bag, position, missing_col), format="csr", dtype=np.float64)
    design.sum_duplicates()
    design.sort_indices()
    if design.shape[0] != len(test) or not np.isfinite(design.data).all():
        raise IntegrityError("Test design is misaligned or nonfinite")
    return design, missing


def infer_candidate(candidate: str) -> dict[str, Any]:
    candidate = candidate.upper()
    freeze = validate_prefit_freeze()
    trained_path, prelabel_path, manifest_path = _candidate_paths(candidate)
    trained = read_signed_json(trained_path)
    prelabel = read_signed_json(prelabel_path)
    if (trained.get("candidate") != candidate
            or trained.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY"
            or trained.get("prefit_freeze_signature") != freeze["signature"]
            or trained.get("input_sha256") != freeze["input_sha256"]
            or prelabel.get("status") != "PRELABEL_PASS"
            or prelabel.get("candidate") != candidate
            or prelabel.get("trained_state_signature") != trained["signature"]
            or prelabel.get("trained_state_sha256") != sha_file(trained_path)
            or prelabel.get("prefit_freeze_signature") != freeze["signature"]
            or prelabel.get("input_sha256") != freeze["input_sha256"]
            or prelabel.get("source_sha256") != freeze["source_sha256"]
            or prelabel.get("test_features_loaded") is not False
            or prelabel.get("test_labels_loaded") is not False):
        raise IntegrityError("Independent prelabel receipt is absent, invalid, or bound to another state")
    state_path = ROOT / trained["candidate_state_path"]
    if sha_file(state_path) != trained["candidate_state_sha256"]:
        raise IntegrityError("Trained-state bundle hash changed")
    if manifest_path.exists():
        raise IntegrityError("Inference manifest exists; inference overwrite/retry forbidden")
    input_sha = current_input_hashes()
    if input_sha != freeze["input_sha256"]:
        raise IntegrityError("Frozen input bytes changed before inference")

    cols = ["record_id", "cycle_code", "signal_stream", "impact_load", "entry_friction",
            "scope_pressure", "text_load", "scope_trace", "org_spread", "module_spread",
            "telemetry_flag", "org_code", "module_code", "pattern_code"]
    test = pd.read_csv(TEST_PATH, usecols=cols, dtype={"record_id": "string"})
    if len(test) != 29421 or test["record_id"].isna().any() or test["record_id"].duplicated().any():
        raise IntegrityError("Official Test feature-only row/ID verification failed")
    cycles = pd.to_numeric(test["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64)):
        raise IntegrityError("Official Test is not complete cycles 80-91")
    sample = pd.read_csv(SAMPLE_PATH, usecols=["record_id"], dtype={"record_id": "string"})
    test_ids = test["record_id"].astype(str).to_numpy(dtype=str)
    if not np.array_equal(test_ids, sample["record_id"].astype(str).to_numpy(dtype=str)):
        raise IntegrityError("Official Test IDs differ from SampleSubmission order")
    base = pd.read_csv(ANCHOR_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    if not np.array_equal(test_ids, base["record_id"].astype(str).to_numpy(dtype=str)):
        raise IntegrityError("Official Test IDs differ from the current R06 final CSV order")
    baseline = pd.to_numeric(base["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100)):
        raise IntegrityError("Current R06 final mapped predictions are invalid")

    state_bundle = joblib.load(state_path)
    model = _load_unit(state_bundle["model_id"])
    design, missing = _infer_design(candidate, test, state_bundle["transform"])
    with threadpool_limits(limits=2):
        raw = np.asarray(model.predict(design), dtype=np.float64)
    if raw.shape != (len(test),) or not np.isfinite(raw).all():
        raise IntegrityError("Candidate Test raw scores are invalid")
    prediction = _candidate_output_prediction(candidate, raw, cycles, missing, baseline)
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100)):
        raise IntegrityError("Candidate output violates [0,100]")

    candidate_dir = CSV_ROOT
    candidate_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "r12_current_route_e79.csv" if candidate == "A" else "direct_position_real_e79.csv"
    csv_path = candidate_dir / csv_name
    raw_path = RUN_ROOT / f"inference_{candidate}.npz"
    if csv_path.exists() or raw_path.exists():
        raise IntegrityError("Candidate inference output exists; overwrite/retry forbidden")
    output = pd.DataFrame({"record_id": test_ids, "threat_rank": prediction})
    output.to_csv(csv_path, index=False, float_format="%.17g", lineterminator="\n")
    with raw_path.open("wb") as stream:
        np.savez_compressed(stream, record_id=test_ids, raw=raw, cycle_code=cycles,
                            missing=missing.astype(bool), baseline=baseline,
                            prediction=prediction)
        stream.flush()
        os.fsync(stream.fileno())
    base_body = {
        "schema_version": 1,
        "namespace": "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1",
        "candidate": candidate,
        "candidate_id": protocol()["candidate_A" if candidate == "A" else "candidate_B"]["id"],
        "status": "CSV_READY_PENDING_INDEPENDENT_QA",
        "scientific_status": "EXPLORATORY_ONLY_PRIOR_LOCAL_GATES_FAILED",
        "trained_state_signature": trained["signature"],
        "prelabel_signature": prelabel["signature"],
        "prefit_freeze_signature": freeze["signature"],
        "input_sha256": input_sha,
        "test_features_loaded_after_trained_state_freeze": True,
        "test_labels_loaded": False,
        "test_rows": int(len(test)),
        "ordered_official_test_ids_sha256": sha_array(test_ids),
        "test_cycle_counts": {str(int(c)): int(np.sum(cycles == c)) for c in np.unique(cycles)},
        "baseline_member_used": "manifest-bound final mapped R06 prediction",
        "baseline_raw_candidate_member_used": False,
        "baseline_csv_sha256": sha_file(ANCHOR_CSV),
        "baseline_prediction_member_digest": protocol()["anchor"]["final_prediction_digest"],
        "raw_prediction_archive": raw_path.relative_to(ROOT).as_posix(),
        "raw_prediction_archive_sha256": sha_file(raw_path),
        "raw_score_sha256": sha_array(raw, dtype="<f8"),
        "missing_rows": int(missing.sum()),
        "missing_fallback_bit_exact": bool(np.array_equal(prediction[missing], baseline[missing])),
        "csv_path": csv_path.relative_to(ROOT).as_posix(),
        "csv_sha256": sha_file(csv_path),
        "prediction_sha256": sha_array(prediction, dtype="<f8"),
        "prediction_range": [float(prediction.min()), float(prediction.max())],
        "prediction_rows": int(len(prediction)),
        "kaggle_accessed": False,
        "upload_performed": False,
    }
    manifest = {**base_body, "signature": canonical_sha(base_body),
                "signature_method": "SHA-256 canonical JSON"}
    write_json_once(manifest_path, manifest)
    return manifest


def _synthetic_e2e() -> dict[str, Any]:
    """Exercise the same frozen formulas and fit accounting without real fits."""
    cycles = np.repeat(np.asarray([80, 81], dtype=np.int64), 6)
    baseline = np.linspace(10.0, 90.0, len(cycles), dtype=np.float64)
    raw = np.asarray([0.4, 0.2, 0.2, 0.8, 0.7, 0.1, 9, 2, 2, 5, 8, 7], dtype=np.float64)
    missing = np.asarray([False, True, False, False, False, False,
                          False, False, True, False, False, False])
    a = _candidate_output_prediction("A", raw, cycles, missing, baseline)
    b = _candidate_output_prediction("B", raw, cycles, missing, baseline)
    if not np.array_equal(a[missing], baseline[missing]) or not np.array_equal(b[missing], baseline[missing]):
        raise IntegrityError("Synthetic fallback is not bit-exact")
    if np.array_equal(a, b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise IntegrityError("Synthetic candidate formulas did not remain distinct and finite")
    plan = protocol()["fit_order_and_cap"]
    # Protocol stores the exact ordered fit IDs as strings.
    if len(plan["candidate_A_order"]) != 5 or len(plan["candidate_B_order"]) != 5:
        raise IntegrityError("Synthetic frozen fit plan is not 5+5 learned units")
    return {
        "status": "SYNTHETIC_ENTRYPOINT_PASS",
        "formula_A_checks": ["full-cycle average ties", "missing exact R06", "range"],
        "formula_B_checks": ["raw plus average-tie percentile", "missing exact R06", "range"],
        "learned_units_plan": {"candidate_A": 5, "candidate_B": 5, "total": 10},
        "production_predictor_fits": 0,
        "official_data_read": False,
        "test_labels_read": False,
        "signature": canonical_sha({"a": a.tolist(), "b": b.tolist(), "cycles": cycles.tolist()}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "freeze", "successor-freeze", "train", "infer"), required=True)
    parser.add_argument("--candidate", choices=("A", "B"))
    args = parser.parse_args(argv)
    if args.mode in {"train", "infer"} and args.candidate is None:
        parser.error("--candidate is required for train/infer")
    if args.mode == "synthetic":
        result = _synthetic_e2e()
    elif args.mode == "freeze":
        result = prefit_freeze()
    elif args.mode == "successor-freeze":
        result = successor_prefit_freeze_v2()
    elif args.mode == "train":
        result = fit_candidate(args.candidate)
    else:
        result = infer_candidate(args.candidate)
    print(json.dumps(to_builtin(result), sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
