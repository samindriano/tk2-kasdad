"""Frozen end-to-end runner for the CLS_DAN_V1 two-origin pilot.

Commands are intentionally separated at the committed prelabel boundary:
synthetic, freeze, prepare, fit-one, predictions, evaluate. No command opens a
Test or SampleSubmission file or contacts Kaggle.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

AUDIT = Path(__file__).resolve().parents[1]
if str(AUDIT) not in sys.path:
    sys.path.insert(0, str(AUDIT))

from src.models import cls_dan_v1 as core
from scripts.verify_cls_dan_v1 import verify_against_sklearn, verify_metric_map


REPORT_ROOT = AUDIT / "reports/simplification_v1/CLS_DAN_V1"
RUN_ROOT = AUDIT / "experiments/CLS_DAN_V1"
FREEZE_PATH = REPORT_ROOT / "SOURCE_INPUT_FREEZE.json"
ROOT_PASS_PATH = REPORT_ROOT / "ROOT_PREFIT_PASS.json"
RESERVATION_PATH = RUN_ROOT / "RESERVATION.json"
FEATURES_READY_PATH = RUN_ROOT / "FEATURES_READY.json"
PREDICTIONS_READY_PATH = RUN_ROOT / "PREDICTIONS_READY.json"
PRELABEL_PATH = RUN_ROOT / "predictions/PRELABEL.npz"
FIT_LEDGER_PATH = RUN_ROOT / "FIT_LEDGER.jsonl"
RESULTS_PATH = RUN_ROOT / "RESULTS.json"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
ORIGINS = (67, 75)
FIT_IDS = {
    "E67_DAN": (67, "DAN"), "E67_LINEAR": (67, "LINEAR"),
    "E75_DAN": (75, "DAN"), "E75_LINEAR": (75, "LINEAR"),
}
SOURCE_FILES = (
    "src/models/cls_dan_v1.py",
    "scripts/cls_dan_v1.py",
    "scripts/verify_cls_dan_v1.py",
    "tests/test_cls_dan_v1.py",
    "reports/simplification_v1/CLS_DAN_V1/PROTOCOL.md",
    "reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_CAMPAIGN.md",
    "reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_VOCAB_PARITY.json",
    "scripts/cls_pos_poisson_bag2_v1.py",
    "src/models/cls_pos_poisson_bag2_v1.py",
)
INPUT_THRESHOLDS = {"start_bytes": 1024**3, "reserve_bytes": 512 * 1024**2,
                    "cache_start_bytes": 512 * 1024**2, "cache_reserve_bytes": 256 * 1024**2,
                    "thread_limit": 2}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha256({"shape": list(array.shape), "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sha256_csr(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256(canonical_sha256({"shape": list(matrix.shape),
                                             "dtype": str(matrix.dtype), "format": "csr"}).encode("ascii"))
    digest.update(np.ascontiguousarray(matrix.indptr, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix.indices, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix.data, dtype=np.float64).tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any], *, write_once: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, indent=2) + "\n").encode("utf-8")
    if write_once:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    require(Path(path).is_file(), f"Required JSON missing: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _load_audit_helper():
    module = importlib.import_module("scripts.cls_pos_poisson_bag2_v1")
    require(Path(module.AUDIT).resolve() == AUDIT.resolve(),
            "Historical evidence helper came from a different checkout")
    return module


def _load_previous_runner(official_root: Path):
    official_root = Path(official_root).resolve()
    sys.path.insert(0, str(official_root))
    importlib.invalidate_caches()
    module = importlib.import_module("scripts.optimization_v3_positional_bag_offset_v1")
    require(Path(module.ROOT).resolve() == official_root,
            "Historical C2_B verifier was loaded from a different official repository")
    return module


def _verify_vocab_parity(official_root: Path, origin: int, manifest: Mapping[str, Any]) -> dict[str, Any]:
    parity = read_json(REPORT_ROOT.parent / "CLS_DUAL_REPRESENTATION_V1/ROOT_VOCAB_PARITY.json")
    require(parity.get("status") == "PASS", "Root unigram vocabulary parity evidence is not PASS")
    record = parity["origins"][str(origin)]
    model_path = Path(record["a02_model_path"])
    require(model_path.is_file() and sha256_file(model_path) == record["model_sha256"],
            f"E{origin} A02 vectorizer artifact hash differs from root parity evidence")
    model = joblib.load(model_path)
    vectorizer = getattr(getattr(model, "text_space", None), "vectorizer", None)
    require(vectorizer is not None and isinstance(vectorizer.vocabulary_, dict),
            f"E{origin} A02 model has no frozen fitted vectorizer")
    helper = _load_previous_runner(official_root)
    vocabulary_hash = helper.canonical_sha256({
        "vocabulary": sorted((str(k), int(v)) for k, v in vectorizer.vocabulary_.items()),
        "idf_sha256": helper.sha256_array(vectorizer.idf_),
    })
    require(vocabulary_hash == record["tfidf_vocab_idf_sha256"]
            == manifest["shared_preprocessing_sha256"]["tfidf_vocabulary_idf"],
            f"E{origin} A02 vocabulary/IDF no longer matches the frozen positional prefix")
    terms, term_to_column = core.unigram_terms(vectorizer)
    require(len(terms) == record["unigram_count"], f"E{origin} unigram count differs from parity")
    require(len(vectorizer.vocabulary_) == record["total_tfidf_features"],
            f"E{origin} full TF-IDF width differs from parity")
    ordered_unigram_pairs = sorted(
        [(str(term), int(index)) for term, index in vectorizer.vocabulary_.items()
         if " " not in str(term)], key=lambda item: item[1]
    )
    ordered_pair_hash = hashlib.sha256(json.dumps(
        ordered_unigram_pairs, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    require(ordered_pair_hash == record["unigram_order_sha256"],
            f"E{origin} ordered unigram/index map differs from root parity evidence")
    base_width = int(manifest["matrices"]["fit_base"]["shape"][1])
    structure_width = base_width - len(vectorizer.vocabulary_) - 1
    require(structure_width == record["structured_width"],
            f"E{origin} structured width differs from frozen C2_B block")
    return {"model_path": str(model_path), "model_sha256": record["model_sha256"],
            "vocabulary_idf_sha256": vocabulary_hash, "unigram_count": len(terms),
            "unigram_order_sha256": ordered_pair_hash,
            "unigram_terms_sha256": sha256_array(np.asarray(terms, dtype=str)),
            "structured_width": structure_width, "tfidf_width": len(vectorizer.vocabulary_),
            "final_missing_indicator_width": 1,
            "tokenizer_rule": "frozen vectorizer preprocessor then tokenizer",
            "unigram_index_rule": "no-space terms ordered by original vectorizer index"}


def _input_file_inventory(official_root: Path, helper: Any,
                          contexts: Mapping[int, Mapping[str, Any]]) -> dict[str, str]:
    root = Path(official_root).resolve()
    paths: set[Path] = {root / "data/raw/classification/Train.csv",
                        Path(helper.RECIPE), Path(helper.ACCOUNTING),
                        Path(helper.PILOT_ROOT) / "FREEZE.json"}
    a02_root = root / "experiments/runs/optimization_v3_a02_neighbors_v1"
    for origin in ORIGINS:
        suffix = f"E{origin}"
        manifest_path = Path(helper.PILOT_ROOT) / f"MATRIX_MANIFEST_{suffix}.json"
        paths.add(manifest_path)
        manifest = read_json(manifest_path)
        for key in ("fit_base", "score_base", "fit_position", "score_position"):
            matrix_path = Path(manifest["matrices"][key]["path"])
            if not matrix_path.is_absolute():
                matrix_path = root / matrix_path
            paths.add(matrix_path)
        paths.add(Path(helper.PILOT_ROOT) / suffix / f"predictions_{suffix}.npz")
        paths.add(a02_root / suffix / "observed/fit_receipt.json")
        paths.add(a02_root / suffix / "observed/predictions.json")
        paths.add(a02_root / suffix / "observed/predictions.npz")
        paths.add(a02_root / suffix / "observed/model.joblib")
        for key in ("gamma_path", "gamma_receipt_path", "pos_predictions_path", "a02_prediction_cache_path"):
            paths.add(Path(contexts[origin][key]))
        accounting = read_json(Path(helper.ACCOUNTING))
        fit_ids = {f"POSITIONAL_TEXT_PILOT_V1_{suffix}_BAG_LR",
                   f"POSITIONAL_TEXT_PILOT_V1_{suffix}_MISSING_HGB",
                   *(f"POSITIONAL_TEXT_PILOT_V1_{suffix}_{component}" for component in
                     ("C2B_STRUCTURED", "C2B_SCALER", "C2B_OHE", "C2B_TFIDF", "C1B_STRUCTURED"))}
        for unit in accounting["fit_units"]:
            if unit["fit_id"] in fit_ids:
                paths.add(Path(helper.PILOT_ROOT) / "units" / f"{unit['fit_id']}.joblib")
                paths.add(Path(helper.PILOT_ROOT) / "receipts" / f"{unit['fit_id']}.json")
    files: dict[str, str] = {}
    for path in sorted(paths, key=lambda p: str(p).lower()):
        resolved = path.resolve()
        require(root == resolved or root in resolved.parents,
                f"Frozen external input escaped the official checkout: {resolved}")
        rel = resolved.relative_to(root).as_posix()
        require("Test.csv" not in rel and "SampleSubmission" not in rel,
                "Test or SampleSubmission is outside this pilot's source boundary")
        require(resolved.is_file(), f"Frozen official input missing: {rel}")
        files[rel] = sha256_file(resolved)
    return files


def _read_input_bundle(official_root: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], Any]:
    official_root = Path(official_root).resolve()
    train = official_root / "data/raw/classification/Train.csv"
    require(train.is_file() and sha256_file(train) == EXPECTED_TRAIN_SHA,
            "Official Classification Train hash does not match the campaign freeze")
    helper = _load_audit_helper()
    old = _load_previous_runner(official_root)
    baseline = old.verify_saved_inputs(include_current_code=True)
    require(baseline["input_sha256"] == EXPECTED_TRAIN_SHA and baseline["input_rows"] == 50608,
            "Historical C2_B/A02/POS read-only verification failed")
    contexts: dict[int, dict[str, Any]] = {}
    vocab_records: dict[str, Any] = {}
    try:
        for origin in ORIGINS:
            context = helper._load_origin_context(official_root, origin, require_fit_labels=False)
            require(context["y_fit"] is None, "Target-free source verification unexpectedly loaded fit labels")
            manifest = context["manifest"]
            vocab_record = _verify_vocab_parity(official_root, origin, manifest)
            require(len(context["fit_rows"]) == (29705 if origin == 67 else 42117),
                    f"E{origin} frozen fit row count mismatch")
            require(len(context["score_rows"]) == (20903 if origin == 67 else 8491),
                    f"E{origin} frozen query row count mismatch")
            require(context["matrices"]["fit_base"].shape[1]
                    == vocab_record["structured_width"] + vocab_record["tfidf_width"] + 1,
                    f"E{origin} saved base matrix is not structured + TF-IDF + missing flag")
            vocab_records[str(origin)] = {
                **vocab_record,
                "matrix_manifest_sha256": context["manifest_sha256"],
                "score_prediction_sha256": sha256_file(context["pilot_prediction_path"])
                if context.get("pilot_prediction_path") else baseline["origins"][f"E{origin}"]["prediction_bundle_sha256"],
                "fit_rows": len(context["fit_rows"]), "query_rows": len(context["score_rows"]),
                "fit_rows_sha256": helper.sha256_array(context["fit_rows"]),
                "fit_record_ids_sha256": helper.sha256_array(context["fit_ids"]),
                "query_rows_sha256": helper.sha256_array(context["score_rows"]),
                "query_record_ids_sha256": helper.sha256_array(context["score_ids"]),
                "fit_prefix_target_sha256_expected": context["fit_target_sha256_expected"],
                "fit_labels_loaded": False,
                "query_labels_loaded": False,
                "present_query_rows": int(context["present"].sum()),
                "missing_query_rows": int((~context["present"]).sum()),
                "champion_present_formula": "0.5 * original_POS + 0.5 * A02",
                "champion_missing_route": "A02 probability, threshold 0.53230976129477",
            }
            contexts[origin] = context
        input_files = _input_file_inventory(official_root, old, contexts)
        local_source_hashes = {rel: sha256_file(AUDIT / rel) for rel in SOURCE_FILES}
        for relative, expected in local_source_hashes.items():
            committed = subprocess.check_output(["git", "-C", str(AUDIT), "show",
                                                 f"HEAD:{relative}"])
            require(hashlib.sha256(committed).hexdigest() == expected,
                    f"Local source file must be committed before freeze: {relative}")
        return {"baseline": baseline, "origins": vocab_records,
                "input_file_hashes": input_files,
                "external_source_hashes": baseline["source_hashes"],
                "local_source_hashes": local_source_hashes,
                "official_root": str(official_root),
                "official_head": git(official_root, "rev-parse", "HEAD")}, contexts, old
    finally:
        for context in contexts.values():
            context.clear()
        contexts.clear()
        gc.collect()


def build_source_input_freeze(official_root: Path) -> dict[str, Any]:
    require(not FREEZE_PATH.exists(), "Source/input freeze is immutable and already exists")
    status = git(AUDIT, "status", "--porcelain=v1")
    require(not status, "Audit checkout must be clean before creating the source/input freeze")
    bundle, _, _ = _read_input_bundle(official_root)
    body = {
        "candidate_id": "CLS_DAN_V1",
        "status": "PASS_SOURCE_INPUT_FREEZE_TARGET_FREE",
        "source_commit": git(AUDIT, "rev-parse", "HEAD"),
        "source_branch": git(AUDIT, "branch", "--show-current"),
        "source_file_hashes": bundle["local_source_hashes"],
        "official_repository_root": bundle["official_root"],
        "official_repository_head_at_freeze": bundle["official_head"],
        "official_train_sha256": EXPECTED_TRAIN_SHA,
        "input_file_hashes": bundle["input_file_hashes"],
        "external_source_hashes": bundle["external_source_hashes"],
        "baseline_verification": bundle["baseline"],
        "origins": bundle["origins"],
        "official_fit_ids": list(FIT_IDS),
        "official_predictor_fit_cap": 4,
        "base_preprocessing_fits": 0,
        "fixed_recipe": {
            "embedding_width": core.EMBEDDING_WIDTH,
            "hidden_widths": list(core.DAN_HIDDEN_WIDTHS),
            "linear_control": "same learned E32 + structured block -> one logit",
            "epochs": core.EPOCHS, "batch_size": core.BATCH_SIZE,
            "learning_rate": core.LEARNING_RATE, "adam_beta1": core.BETA1,
            "adam_beta2": core.BETA2, "adam_epsilon": core.ADAM_EPSILON,
            "dropout": core.DROPOUT_RATE, "l2": core.L2_COEFFICIENT,
            "seed": core.RANDOM_SEED, "dtype": "float64",
        },
        "blend": {"present": "0.75 * champion + 0.25 * auxiliary",
                  "missing": "exact champion fallback", "present_threshold": 0.50,
                  "missing_threshold": 0.53230976129477},
        "data_boundary": {"train_only": True, "fit_prefix_labels_read": False,
                          "query_labels_read": False, "test_opened": False,
                          "sample_submission_opened": False, "kaggle_contacted": False},
        "resource_policy": INPUT_THRESHOLDS,
        "retry_policy": "failed/interrupted predictor starts consume their slot; no retry",
    }
    body["freeze_signature"] = canonical_sha256(body)
    atomic_json(FREEZE_PATH, body, write_once=True)
    return body


def verify_frozen_sources(official_root: Path) -> dict[str, Any]:
    freeze = read_json(FREEZE_PATH)
    body = dict(freeze)
    signature = body.pop("freeze_signature", None)
    require(signature == canonical_sha256(body), "Source/input freeze signature mismatch")
    require(freeze["candidate_id"] == "CLS_DAN_V1"
            and freeze["official_repository_root"] == str(Path(official_root).resolve()),
            "Frozen candidate or official repository identity mismatch")
    freeze_rel = FREEZE_PATH.relative_to(AUDIT).as_posix()
    committed = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"HEAD:{freeze_rel}"])
    require(committed.replace(b"\r\n", b"\n") == FREEZE_PATH.read_bytes().replace(b"\r\n", b"\n"),
            "Source/input freeze is not committed exactly")
    source_commit = freeze["source_commit"]
    require(git(AUDIT, "merge-base", "--is-ancestor", source_commit, git(AUDIT, "rev-parse", "HEAD")) == "",
            "Frozen source commit is not an ancestor of current HEAD")
    for relative, expected in freeze["source_file_hashes"].items():
        require(sha256_file(AUDIT / relative) == expected,
                f"Frozen local source file changed: {relative}")
        blob = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"{source_commit}:{relative}"])
        require(hashlib.sha256(blob).hexdigest() == expected,
                f"Committed frozen source blob differs: {relative}")
    root = Path(official_root).resolve()
    for relative, expected in {**freeze["external_source_hashes"], **freeze["input_file_hashes"]}.items():
        require(sha256_file(root / relative) == expected,
                f"Official frozen input/source changed: {relative}")
    require(freeze["official_train_sha256"] == EXPECTED_TRAIN_SHA
            and not any("Test.csv" in p or "SampleSubmission" in p for p in freeze["input_file_hashes"]),
            "Official data boundary differs from the source freeze")
    return freeze


def _require_committed_json(path: Path) -> dict[str, Any]:
    relative = Path(path).resolve().relative_to(AUDIT.resolve()).as_posix()
    committed = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"HEAD:{relative}"])
    on_disk = Path(path).read_bytes()
    require(committed.replace(b"\r\n", b"\n") == on_disk.replace(b"\r\n", b"\n"),
            f"Required approval/evidence is not committed exactly: {relative}")
    return json.loads(on_disk.decode("utf-8"))


def _require_committed_file(path: Path) -> None:
    relative = Path(path).resolve().relative_to(AUDIT.resolve()).as_posix()
    committed = subprocess.check_output(["git", "-C", str(AUDIT), "show", f"HEAD:{relative}"])
    require(hashlib.sha256(committed).hexdigest() == sha256_file(path),
            f"Required prelabel artifact is not committed byte-for-byte: {relative}")


def verify_root_prefit_and_reservation(official_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = verify_frozen_sources(official_root)
    approval = _require_committed_json(ROOT_PASS_PATH)
    require(approval.get("status") == "PASS_ROOT_PREFIT_APPROVED"
            and approval.get("candidate_id") == "CLS_DAN_V1"
            and approval.get("freeze_signature") == freeze["freeze_signature"]
            and approval.get("source_commit") == freeze["source_commit"]
            and approval.get("fit_ids") == list(FIT_IDS)
            and approval.get("fit_cap") == 4
            and approval.get("test_or_sample_access") is False
            and approval.get("kaggle_upload") is False,
            "Committed root prefit pass does not match the frozen four-fit lane")
    reservation = _require_committed_json(RESERVATION_PATH)
    require(reservation.get("status") == "RESERVED"
            and reservation.get("candidate_id") == "CLS_DAN_V1"
            and reservation.get("freeze_signature") == freeze["freeze_signature"]
            and reservation.get("fit_ids") == list(FIT_IDS)
            and reservation.get("fit_start_cap") == 4
            and reservation.get("no_retry") is True,
            "Committed four-fit reservation does not match the source freeze")
    return freeze, approval, reservation


@contextmanager
def _os_lock(path: Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
        os.fsync(stream.fileno())
    stream.seek(0)
    acquired = False
    unlock = None
    try:
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Shared/local advisory lock is held: {path}") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Shared/local advisory lock is held: {path}") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        acquired = True
        yield {"path": str(path), "acquired": True, "mechanism": "OS advisory byte lock"}
    finally:
        if acquired and unlock is not None:
            try:
                unlock()
            except OSError:
                pass
        stream.close()


@contextmanager
def heavy_job_locks() -> Iterator[list[dict[str, Any]]]:
    with ExitStack() as stack:
        shared = stack.enter_context(_os_lock(Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"))
        local = stack.enter_context(_os_lock(RUN_ROOT / ".training.lock"))
        yield [shared, local]


def _resource_snapshot(start_bytes: int, reserve_bytes: int) -> dict[str, int]:
    memory = psutil.virtual_memory()
    require(memory.available >= start_bytes,
            f"Available memory {memory.available} is below start gate {start_bytes}")
    return {"available_bytes": int(memory.available), "total_bytes": int(memory.total),
            "start_gate_bytes": int(start_bytes), "postfit_reserve_bytes": int(reserve_bytes)}


def run_synthetic_lifecycle(output_dir: Path) -> dict[str, Any]:
    """Exercise fit, predict, persistence, readback, and metrics on synthetic rows only."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not any(output_dir.iterdir()), "Synthetic output directory must be empty")
    training_docs = [
        "alpha alpha beta" if i % 2 == 0 else "gamma delta gamma"
        for i in range(64)
    ]
    query_docs = ["alpha beta", "gamma delta", "alpha alpha", "delta gamma"] * 4
    y_train = np.asarray([0 if i % 2 == 0 else 1 for i in range(64)], dtype=np.float64)
    y_query = np.asarray([0, 1, 0, 1] * 4, dtype=np.int8)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    vectorizer.fit(training_docs)
    counts_fit, fit_audit = core.build_mean_unigram_counts(training_docs, vectorizer)
    counts_query, query_audit = core.build_mean_unigram_counts(query_docs, vectorizer)
    rng = np.random.default_rng(91)
    structured_fit = rng.normal(0.0, 0.5, size=(len(training_docs), 3)).astype(np.float64)
    structured_query = rng.normal(0.0, 0.5, size=(len(query_docs), 3)).astype(np.float64)
    verification: dict[str, Any] = {}
    max_diff = 0.0
    model_records: dict[str, Any] = {}
    for architecture in core.ARCHITECTURES:
        model = core.fit_fixed_recipe(counts_fit, structured_fit, y_train, architecture)
        predictions = core.predict_probabilities(model, counts_query, structured_query)
        model_path = output_dir / f"synthetic_{architecture}.npz"
        atomic_npz(model_path, **core.model_arrays(model))
        with np.load(model_path, allow_pickle=False) as archive:
            restored = core.model_from_arrays({name: archive[name] for name in archive.files})
        replay = core.predict_probabilities(restored, counts_query, structured_query)
        max_diff = max(max_diff, float(np.max(np.abs(predictions - replay))))
        verification[architecture] = verify_against_sklearn(y_query, replay, np.ones(len(y_query), dtype=bool))
        model_records[architecture] = {
            "artifact_sha256": sha256_file(model_path), "updates": model["adam_steps"],
            "train_loss_initial": model["training"]["initial_no_dropout_loss"],
            "train_loss_final": model["training"]["final_no_dropout_loss"],
            "query_probability_sha256": sha256_array(replay),
            "fit_count_coverage_mean": float(np.mean(fit_audit["coverage"])),
            "query_count_coverage_mean": float(np.mean(query_audit["coverage"])),
        }
    require(max_diff == 0.0, "Synthetic save/reload did not preserve predictions bit-for-bit")
    require(all(item["status"] == "PASS" for item in verification.values()),
            "Synthetic independent metric verifier failed")
    receipt = {
        "status": "PASS_SYNTHETIC_ONLY",
        "candidate_id": "CLS_DAN_V1",
        "fit_type": "SYNTHETIC_ONLY",
        "official_data_opened": False,
        "official_fit_starts": 0,
        "query_targets_used_for_synthetic_metric_only": True,
        "vectorizer_fit_scope": "synthetic training strings only",
        "models": model_records,
        "readback_probability_max_abs_diff": max_diff,
        "metric_verifier_match": True,
        "synthetic_metrics": verification,
    }
    atomic_json(output_dir / "SYNTHETIC_QA.json", receipt, write_once=True)
    return receipt


def _normalize_streams(values: Sequence[Any]) -> tuple[list[str], np.ndarray]:
    documents: list[str] = []
    missing = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        if pd.isna(value):
            documents.append("")
            missing[index] = True
        else:
            text = str(value).strip()
            documents.append(text)
            missing[index] = not bool(text)
    return documents, missing


def _csr_payload(prefix: str, matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    csr = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    return {
        f"{prefix}_data": np.asarray(csr.data, dtype=np.float64),
        f"{prefix}_indices": np.asarray(csr.indices, dtype=np.int32),
        f"{prefix}_indptr": np.asarray(csr.indptr, dtype=np.int64),
        f"{prefix}_shape": np.asarray(csr.shape, dtype=np.int64),
    }


def _csr_from_payload(archive: Mapping[str, Any], prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(x) for x in np.asarray(archive[f"{prefix}_shape"], dtype=np.int64))
    matrix = sparse.csr_matrix(
        (np.asarray(archive[f"{prefix}_data"], dtype=np.float64),
         np.asarray(archive[f"{prefix}_indices"], dtype=np.int32),
         np.asarray(archive[f"{prefix}_indptr"], dtype=np.int64)),
        shape=shape, dtype=np.float64,
    )
    require(np.isfinite(matrix.data).all(), f"Feature cache {prefix} contains non-finite values")
    return matrix


def _normal_group(values: Any) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        if pd.isna(value):
            result.append("<MISSING>")
        else:
            result.append(str(value))
    return np.asarray(result, dtype=str)


def prepare_features(official_root: Path) -> dict[str, Any]:
    """Build frozen-vocabulary mean-count/structured caches; no labels read."""
    freeze, _, reservation = verify_root_prefit_and_reservation(official_root)
    require(not FEATURES_READY_PATH.exists(), "Feature cache preparation is write-once")
    with heavy_job_locks():
        resource_start = _resource_snapshot(INPUT_THRESHOLDS["cache_start_bytes"],
                                            INPUT_THRESHOLDS["cache_reserve_bytes"])
        official_root = Path(official_root).resolve()
        train_path = official_root / "data/raw/classification/Train.csv"
        data = pd.read_csv(train_path, usecols=["record_id", "cycle_code", "signal_stream"],
                           dtype={"record_id": "string"})
        require(len(data) == 50608 and data["record_id"].is_unique
                and not data["record_id"].isna().any(),
                "Official Train identity schema/count changed during feature preparation")
        cycles = pd.to_numeric(data["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        require(int(cycles.min()) == 47 and int(cycles.max()) == 79,
                "Official Train cycle range changed")
        documents, raw_missing = _normalize_streams(data["signal_stream"].to_numpy(dtype=object))
        helper = _load_audit_helper()
        ready_origins: dict[str, Any] = {}
        try:
            for origin in ORIGINS:
                context = helper._load_origin_context(official_root, origin, require_fit_labels=False)
                manifest = context["manifest"]
                parity = _verify_vocab_parity(official_root, origin, manifest)
                fit_rows = np.asarray(context["fit_rows"], dtype=np.int64)
                score_rows = np.asarray(context["score_rows"], dtype=np.int64)
                fit_ids = data.iloc[fit_rows]["record_id"].astype(str).to_numpy(dtype=str)
                score_ids = data.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
                require(np.array_equal(fit_ids, context["fit_ids"])
                        and np.array_equal(score_ids, context["score_ids"]),
                        f"E{origin} feature source IDs differ from the frozen matrix rows")
                require(np.array_equal(cycles[fit_rows],
                                       np.asarray(context["manifest"]["fit_row_metadata"]["cycles"], dtype=np.int64))
                        if "cycles" in context["manifest"]["fit_row_metadata"] else
                        np.all((cycles[fit_rows] >= 47) & (cycles[fit_rows] <= origin)),
                        f"E{origin} fit rows escape their source cycle prefix")
                fit_base = sparse.csr_matrix(context["matrices"]["fit_base"], dtype=np.float64)
                score_base = sparse.csr_matrix(context["matrices"]["score_base"], dtype=np.float64)
                width = int(parity["structured_width"])
                fit_missing_saved = np.asarray(fit_base[:, -1].toarray()).reshape(-1).astype(bool)
                score_missing_saved = np.asarray(score_base[:, -1].toarray()).reshape(-1).astype(bool)
                require(np.array_equal(fit_missing_saved, raw_missing[fit_rows])
                        and np.array_equal(score_missing_saved, raw_missing[score_rows]),
                        f"E{origin} saved stream-missing column differs from raw Train features")
                fit_present_positions = np.flatnonzero(~fit_missing_saved).astype(np.int64)
                score_present_positions = np.flatnonzero(~score_missing_saved).astype(np.int64)
                require(len(fit_present_positions) > 0 and len(score_present_positions) > 0,
                        f"E{origin} has no present-stream rows for the auxiliary route")

                model_path = Path(parity["model_path"])
                a02_model = joblib.load(model_path)
                vectorizer = a02_model.text_space.vectorizer
                fit_docs = [documents[row] for row in fit_rows[fit_present_positions]]
                score_docs = [documents[row] for row in score_rows[score_present_positions]]
                fit_counts, fit_audit = core.build_mean_unigram_counts(fit_docs, vectorizer)
                score_counts, score_audit = core.build_mean_unigram_counts(score_docs, vectorizer)
                require(fit_counts.shape[1] == score_counts.shape[1] == parity["unigram_count"],
                        f"E{origin} mean-count vocabulary width differs from the frozen map")
                fit_structured = np.asarray(fit_base[:, :width][fit_present_positions].toarray(), dtype=np.float64)
                score_structured = np.asarray(score_base[:, :width][score_present_positions].toarray(), dtype=np.float64)
                require(np.isfinite(fit_structured).all() and np.isfinite(score_structured).all()
                        and fit_structured.shape[1] == score_structured.shape[1] == width,
                        f"E{origin} structured feature slice is invalid")
                present = np.asarray(context["present"], dtype=bool)
                require(np.array_equal(present, ~score_missing_saved),
                        f"E{origin} original champion availability differs from the saved C2_B missing flag")
                champion = np.asarray(context["p_a02"], dtype=np.float64).copy()
                champion[present] = 0.5 * np.asarray(context["p_pos_original"], dtype=np.float64)[present] + 0.5 * champion[present]
                require(np.isfinite(champion).all()
                        and np.array_equal(champion[~present].view(np.uint64),
                                           np.asarray(context["p_a02"], dtype=np.float64)[~present].view(np.uint64)),
                        f"E{origin} champion missing-stream route is not an exact A02 fallback")

                payload: dict[str, Any] = {}
                payload.update(_csr_payload("fit_counts", fit_counts))
                payload.update(_csr_payload("score_counts", score_counts))
                payload.update({
                    "fit_structured": fit_structured,
                    "score_structured": score_structured,
                    "fit_rows": fit_rows,
                    "fit_record_ids": fit_ids,
                    "fit_cycles": cycles[fit_rows],
                    "fit_present_positions": fit_present_positions,
                    "score_rows": score_rows,
                    "score_record_ids": score_ids,
                    "score_cycles": cycles[score_rows],
                    "score_present_positions": score_present_positions,
                    "score_present_mask": present,
                    "score_champion_probability": champion,
                    "score_a02_probability": np.asarray(context["p_a02"], dtype=np.float64),
                    "score_pos_original_probability": np.asarray(context["p_pos_original"], dtype=np.float64),
                    "score_entry_friction": _normal_group(context["entry_friction"]),
                    "score_text_load_group": _normal_group(context["text_load_group"]),
                    "fit_coverage": np.asarray(fit_audit["coverage"], dtype=np.float64),
                    "score_coverage": np.asarray(score_audit["coverage"], dtype=np.float64),
                    "metadata_json": np.asarray(json.dumps({
                        "origin": origin, "freeze_signature": freeze["freeze_signature"],
                        "reservation_sha256": sha256_file(RESERVATION_PATH),
                        "matrix_manifest_sha256": context["manifest_sha256"],
                        "fit_target_sha256_expected": context["fit_target_sha256_expected"],
                        "fit_count_rows": int(len(fit_present_positions)),
                        "score_present_rows": int(len(score_present_positions)),
                        "structured_width": width,
                        "unigram_vocab_size": parity["unigram_count"],
                        "base_feature_order": "saved C2_B structured numeric/OHE; excludes TF-IDF and final missing flag",
                        "tokenization": parity["tokenizer_rule"],
                        "fit_labels_read": False, "query_labels_read": False,
                    }, sort_keys=True, ensure_ascii=False, allow_nan=False)),
                })
                path = RUN_ROOT / "features" / f"E{origin}.npz"
                atomic_npz(path, **payload)
                ready_origins[str(origin)] = {
                    "path": path.relative_to(AUDIT).as_posix(),
                    "sha256": sha256_file(path),
                    "fit_rows": len(fit_rows), "fit_present_rows": len(fit_present_positions),
                    "fit_missing_rows": len(fit_rows) - len(fit_present_positions),
                    "fit_cycle_counts": {str(cycle): int(np.count_nonzero(cycles[fit_rows] == cycle))
                                         for cycle in sorted(np.unique(cycles[fit_rows]).tolist())},
                    "fit_label_overlap_with_E67_query_cycles_68_75_rows": int(
                        np.count_nonzero((cycles[fit_rows] >= 68) & (cycles[fit_rows] <= 75))
                    ),
                    "query_rows": len(score_rows), "query_present_rows": len(score_present_positions),
                    "query_missing_rows": len(score_rows) - len(score_present_positions),
                    "fit_rows_sha256": sha256_array(fit_rows),
                    "query_rows_sha256": sha256_array(score_rows),
                    "fit_count_csr_sha256": sha256_csr(fit_counts),
                    "query_count_csr_sha256": sha256_csr(score_counts),
                    "fit_structured_sha256": sha256_array(fit_structured),
                    "query_structured_sha256": sha256_array(score_structured),
                    "fit_token_coverage": {
                        "mean": float(np.mean(fit_audit["coverage"])),
                        "median": float(np.median(fit_audit["coverage"])),
                        "all_oov_rows": fit_audit["all_oov_rows"],
                        "empty_token_rows": fit_audit["empty_token_rows"],
                    },
                    "query_token_coverage": {
                        "mean": float(np.mean(score_audit["coverage"])),
                        "median": float(np.median(score_audit["coverage"])),
                        "all_oov_rows": score_audit["all_oov_rows"],
                        "empty_token_rows": score_audit["empty_token_rows"],
                    },
                    "fit_labels_read": False, "query_labels_read": False,
                }
                context.clear()
                del fit_base, score_base, fit_counts, score_counts, fit_structured, score_structured, payload, a02_model
                gc.collect()
            result = {
                "candidate_id": "CLS_DAN_V1", "status": "PASS_FEATURES_READY_TARGET_FREE",
                "freeze_signature": freeze["freeze_signature"],
                "reservation_sha256": sha256_file(RESERVATION_PATH),
                "official_train_sha256": EXPECTED_TRAIN_SHA,
                "origins": ready_origins,
                "fit_labels_read": False, "query_labels_read": False,
                "test_opened": False, "sample_submission_opened": False,
                "resource_start": resource_start,
                "resource_after": _resource_snapshot(0, INPUT_THRESHOLDS["cache_reserve_bytes"]),
            }
            atomic_json(FEATURES_READY_PATH, result, write_once=True)
            return result
        finally:
            del data, documents, raw_missing
            gc.collect()


def verify_features_ready(official_root: Path, *, freeze: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if freeze is None:
        frozen, _, reservation = verify_root_prefit_and_reservation(official_root)
    else:
        frozen = dict(freeze)
        reservation = _require_committed_json(RESERVATION_PATH)
    ready = _require_committed_json(FEATURES_READY_PATH)
    require(ready.get("status") == "PASS_FEATURES_READY_TARGET_FREE"
            and ready.get("freeze_signature") == frozen["freeze_signature"]
            and ready.get("reservation_sha256") == sha256_file(RESERVATION_PATH)
            and set(ready.get("origins", {})) == {"67", "75"}
            and ready.get("fit_labels_read") is False and ready.get("query_labels_read") is False,
            "Committed target-free feature cache receipt is invalid")
    for origin, record in ready["origins"].items():
        path = AUDIT / record["path"]
        require(path.is_file() and sha256_file(path) == record["sha256"],
                f"E{origin} target-free feature cache is missing or changed")
    return ready, reservation


def _read_exact_targets(train_path: Path, rows: Sequence[int], record_ids: Sequence[str],
                        purpose: str) -> np.ndarray:
    """Read target text only for an explicitly frozen row/ID set."""
    row_array = np.asarray(rows, dtype=np.int64).reshape(-1)
    expected_ids = np.asarray(record_ids).astype(str).reshape(-1)
    require(len(row_array) > 0 and len(row_array) == len(expected_ids)
            and len(np.unique(row_array)) == len(row_array)
            and np.all(row_array >= 0), "Exact target reader requires unique nonnegative row positions")
    row_to_output = {int(row): i for i, row in enumerate(row_array)}
    output = np.empty(len(row_array), dtype=np.int8)
    seen = np.zeros(len(row_array), dtype=bool)
    with Path(train_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        require(header is not None and "record_id" in header and "artifact_signal" in header,
                "Official Train is missing target/identity columns")
        id_column, target_column = header.index("record_id"), header.index("artifact_signal")
        max_row = int(row_array.max())
        for row_position, record in enumerate(reader):
            output_position = row_to_output.get(row_position)
            if output_position is not None:
                require(max(id_column, target_column) < len(record),
                        "Official Train row is shorter than the frozen target schema")
                require(record[id_column] == expected_ids[output_position],
                        f"Target row identity mismatch during {purpose}")
                try:
                    value = int(record[target_column])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"Non-integer binary target during {purpose}") from exc
                require(value in (0, 1), f"Nonbinary target during {purpose}")
                output[output_position] = value
                seen[output_position] = True
            if row_position >= max_row:
                break
    require(seen.all(), f"Exact target reader did not reach every frozen row for {purpose}")
    return output


def _append_ledger(event: Mapping[str, Any]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    with FIT_LEDGER_PATH.open("a", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_events() -> list[dict[str, Any]]:
    if not FIT_LEDGER_PATH.is_file():
        return []
    output = []
    for line_no, line in enumerate(FIT_LEDGER_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            event = json.loads(line)
            require(isinstance(event, dict) and "event" in event and "fit_id" in event,
                    f"Malformed fit ledger at line {line_no}")
            output.append(event)
    return output


def fit_one(official_root: Path, fit_id: str) -> dict[str, Any]:
    """Consume exactly one reserved official prefix-only optimizer start."""
    require(fit_id in FIT_IDS, f"Unreserved official fit ID: {fit_id}")
    freeze, _, reservation = verify_root_prefit_and_reservation(official_root)
    ready, _ = verify_features_ready(official_root, freeze=freeze)
    origin, architecture = FIT_IDS[fit_id]
    cache_record = ready["origins"][str(origin)]
    feature_path = AUDIT / cache_record["path"]
    model_path = RUN_ROOT / "units" / f"{fit_id}.npz"
    prediction_path = RUN_ROOT / "predictions" / f"{fit_id}.npz"
    receipt_path = RUN_ROOT / "units" / "receipts" / f"{fit_id}.json"
    for path in (model_path, prediction_path, receipt_path):
        require(not path.exists(), f"Fit artifact already exists; no retry allowed: {path}")

    started = False
    start_time = time.perf_counter()
    start_resource: dict[str, Any] | None = None
    try:
        with heavy_job_locks(), threadpool_limits(limits=INPUT_THRESHOLDS["thread_limit"]):
            start_resource = _resource_snapshot(INPUT_THRESHOLDS["start_bytes"],
                                                INPUT_THRESHOLDS["reserve_bytes"])
            events = _ledger_events()
            require(not any(row["fit_id"] == fit_id and row["event"] in
                            {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED_CONSUMED"}
                            for row in events),
                    f"Fit slot {fit_id} was already started/consumed; retry is forbidden")
            start_event = {
                "event": "FIT_STARTED", "fit_id": fit_id, "origin": origin,
                "architecture": architecture, "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "freeze_signature": freeze["freeze_signature"],
                "reservation_sha256": sha256_file(RESERVATION_PATH),
                "feature_cache_sha256": cache_record["sha256"],
                "resource_start": start_resource,
                "query_labels_read": False, "test_opened": False,
            }
            _append_ledger(start_event)
            started = True
            with np.load(feature_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                require(metadata["origin"] == origin
                        and metadata["freeze_signature"] == freeze["freeze_signature"]
                        and metadata["fit_labels_read"] is False
                        and metadata["query_labels_read"] is False,
                        "Feature cache metadata no longer matches the frozen target-free state")
                counts = _csr_from_payload(archive, "fit_counts")
                structured = np.asarray(archive["fit_structured"], dtype=np.float64).copy()
                fit_rows = np.asarray(archive["fit_rows"], dtype=np.int64).copy()
                fit_ids = np.asarray(archive["fit_record_ids"]).astype(str)
                present_positions = np.asarray(archive["fit_present_positions"], dtype=np.int64).copy()
                fit_coverage = np.asarray(archive["fit_coverage"], dtype=np.float64).copy()
            y_full = _read_exact_targets(
                Path(official_root) / "data/raw/classification/Train.csv",
                fit_rows, fit_ids, purpose=f"{fit_id} prefix-only fit",
            )
            target_hash = sha256_array(y_full)
            require(target_hash == metadata["fit_target_sha256_expected"],
                    f"{fit_id} prefix target hash differs from the saved C2_B fit receipt")
            y_present = y_full[present_positions].astype(np.float64)
            require(len(y_present) == counts.shape[0] == len(structured)
                    and np.unique(y_present).size == 2,
                    f"{fit_id} present-stream fit rows/labels are invalid")
            fit_started_at = time.perf_counter()
            model = core.fit_fixed_recipe(counts, structured, y_present, architecture)
            fit_seconds = time.perf_counter() - fit_started_at
            with np.load(feature_path, allow_pickle=False) as archive:
                score_counts = _csr_from_payload(archive, "score_counts")
                score_structured = np.asarray(archive["score_structured"], dtype=np.float64).copy()
                score_present_positions = np.asarray(archive["score_present_positions"], dtype=np.int64).copy()
                score_rows = np.asarray(archive["score_rows"], dtype=np.int64).copy()
                score_ids = np.asarray(archive["score_record_ids"]).astype(str)
                score_cycles = np.asarray(archive["score_cycles"], dtype=np.int64).copy()
            score_aux_probability = core.predict_probabilities(model, score_counts, score_structured)
            require(len(score_aux_probability) == len(score_present_positions)
                    and np.isfinite(score_aux_probability).all(),
                    f"{fit_id} query auxiliary probabilities are invalid")

            model_payload = core.model_arrays(model)
            atomic_npz(model_path, **model_payload)
            full_aux = np.full(len(score_rows), np.nan, dtype=np.float64)
            full_aux[score_present_positions] = score_aux_probability
            atomic_npz(prediction_path,
                       score_aux_probability=score_aux_probability,
                       score_present_positions=score_present_positions,
                       score_rows=score_rows, score_record_ids=score_ids,
                       score_cycles=score_cycles, score_aux_sha256=np.asarray(sha256_array(score_aux_probability)))
            after_resource = _resource_snapshot(0, INPUT_THRESHOLDS["reserve_bytes"])
            parameter_hashes = {name: sha256_array(value) for name, value in model["params"].items()}
            fit_receipt = {
                "status": "FIT_SAVED", "candidate_id": "CLS_DAN_V1", "fit_id": fit_id,
                "origin": origin, "architecture": architecture,
                "source_commit": freeze["source_commit"],
                "freeze_signature": freeze["freeze_signature"],
                "reservation_sha256": sha256_file(RESERVATION_PATH),
                "feature_cache_sha256": cache_record["sha256"],
                "model_artifact_path": model_path.relative_to(AUDIT).as_posix(),
                "model_artifact_sha256": sha256_file(model_path),
                "prediction_artifact_path": prediction_path.relative_to(AUDIT).as_posix(),
                "prediction_artifact_sha256": sha256_file(prediction_path),
                "parameter_sha256": parameter_hashes,
                "fit_rows_all_prefix": int(len(fit_rows)),
                "fit_rows_present": int(len(present_positions)),
                "fit_rows_missing_excluded": int(len(fit_rows) - len(present_positions)),
                "fit_record_ids_sha256": sha256_array(fit_ids),
                "fit_row_positions_sha256": sha256_array(fit_rows),
                "fit_target_sha256_verified": target_hash,
                "fit_target_values_read": int(len(y_full)),
                "fit_positive_count_all_prefix": int(y_full.sum()),
                "fit_positive_count_present": int(y_present.sum()),
                "fit_present_token_coverage_mean": float(np.mean(fit_coverage)),
                "query_rows_all": int(len(score_rows)),
                "query_rows_present": int(len(score_present_positions)),
                "query_score_rows_sha256": sha256_array(score_rows),
                "query_score_ids_sha256": sha256_array(score_ids),
                "query_score_cycles_sha256": sha256_array(score_cycles),
                "origin_query_targets_read_for_evaluation": False,
                "test_or_sample_opened": False,
                "kaggle_contacted": False,
                "training": model["training"],
                "optimizer_seconds": fit_seconds,
                "total_elapsed_seconds": time.perf_counter() - start_time,
                "resource_start": start_resource,
                "resource_after": after_resource,
                "resource_reserve_pass": after_resource["available_bytes"] >= INPUT_THRESHOLDS["reserve_bytes"],
                "adam_buffers_persisted": True,
                "no_retry": True,
            }
            atomic_json(receipt_path, fit_receipt, write_once=True)
            _append_ledger({
                "event": "FIT_SAVED", "fit_id": fit_id,
                "receipt_sha256": sha256_file(receipt_path),
                "model_artifact_sha256": sha256_file(model_path),
                "prediction_artifact_sha256": sha256_file(prediction_path),
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            })
            return fit_receipt
    except BaseException as exc:
        if started:
            _append_ledger({
                "event": "FIT_FAILED_CONSUMED", "fit_id": fit_id,
                "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "exception_type": type(exc).__name__, "error": str(exc)[:2000],
                "no_retry": True,
            })
        raise


def build_prelabel_predictions(official_root: Path) -> dict[str, Any]:
    """Read back all four fit states and persist predictions without query labels."""
    freeze, _, _ = verify_root_prefit_and_reservation(official_root)
    ready, _ = verify_features_ready(official_root, freeze=freeze)
    require(not PRELABEL_PATH.exists() and not PREDICTIONS_READY_PATH.exists(),
            "Prelabel predictions are write-once")
    events = _ledger_events()
    saved_ids = {row["fit_id"] for row in events if row["event"] == "FIT_SAVED"}
    require(saved_ids == set(FIT_IDS), "Exactly the four reserved fits must be saved before predictions")
    data_by_origin: dict[int, dict[str, Any]] = {}
    fit_receipt_hashes: dict[str, str] = {}
    fit_artifact_hashes: dict[str, str] = {}
    for fit_id, (origin, architecture) in FIT_IDS.items():
        receipt_path = RUN_ROOT / "units" / "receipts" / f"{fit_id}.json"
        model_path = RUN_ROOT / "units" / f"{fit_id}.npz"
        pred_path = RUN_ROOT / "predictions" / f"{fit_id}.npz"
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "FIT_SAVED" and receipt.get("fit_id") == fit_id
                and receipt.get("architecture") == architecture and receipt.get("origin") == origin
                and receipt.get("freeze_signature") == freeze["freeze_signature"]
                and receipt.get("origin_query_targets_read_for_evaluation") is False,
                f"{fit_id} receipt does not match the frozen fit")
        require(sha256_file(model_path) == receipt["model_artifact_sha256"]
                and sha256_file(pred_path) == receipt["prediction_artifact_sha256"],
                f"{fit_id} fitted artifact hash differs from its receipt")
        feature_path = AUDIT / ready["origins"][str(origin)]["path"]
        with np.load(model_path, allow_pickle=False) as archive:
            model = core.model_from_arrays({name: archive[name] for name in archive.files})
        with np.load(feature_path, allow_pickle=False) as archive:
            score_counts = _csr_from_payload(archive, "score_counts")
            score_structured = np.asarray(archive["score_structured"], dtype=np.float64).copy()
            score_rows = np.asarray(archive["score_rows"], dtype=np.int64).copy()
            score_ids = np.asarray(archive["score_record_ids"]).astype(str)
            score_cycles = np.asarray(archive["score_cycles"], dtype=np.int64).copy()
            present = np.asarray(archive["score_present_mask"], dtype=bool).copy()
            score_present_positions = np.asarray(archive["score_present_positions"], dtype=np.int64).copy()
            champion = np.asarray(archive["score_champion_probability"], dtype=np.float64).copy()
            p_pos = np.asarray(archive["score_pos_original_probability"], dtype=np.float64).copy()
            p_a02 = np.asarray(archive["score_a02_probability"], dtype=np.float64).copy()
            entry_friction = np.asarray(archive["score_entry_friction"]).astype(str)
            text_load_group = np.asarray(archive["score_text_load_group"]).astype(str)
            coverage = np.asarray(archive["score_coverage"], dtype=np.float64).copy()
        with np.load(pred_path, allow_pickle=False) as archive:
            aux_probability = np.asarray(archive["score_aux_probability"], dtype=np.float64).copy()
            saved_present_positions = np.asarray(archive["score_present_positions"], dtype=np.int64).copy()
            require(np.array_equal(np.asarray(archive["score_rows"], dtype=np.int64), score_rows)
                    and np.array_equal(np.asarray(archive["score_record_ids"]).astype(str), score_ids)
                    and np.array_equal(np.asarray(archive["score_cycles"], dtype=np.int64), score_cycles),
                    f"{fit_id} prediction row identities differ from the frozen cache")
        require(np.array_equal(saved_present_positions, score_present_positions)
                and len(aux_probability) == len(score_present_positions),
                f"{fit_id} prediction availability rows differ from the cache")
        replay = core.predict_probabilities(model, score_counts, score_structured)
        require(np.array_equal(aux_probability.view(np.uint64), replay.view(np.uint64)),
                f"{fit_id} saved model readback does not reproduce its query predictions bit-for-bit")
        full_aux = np.full(len(score_rows), np.nan, dtype=np.float64)
        full_aux[score_present_positions] = aux_probability
        candidate = champion.copy()
        candidate[present] = 0.75 * champion[present] + 0.25 * full_aux[present]
        require(np.isfinite(candidate).all()
                and np.array_equal(candidate[~present].view(np.uint64), champion[~present].view(np.uint64)),
                f"{fit_id} candidate probability route is not finite/exact on missing rows")
        origin_record = data_by_origin.setdefault(origin, {
            "score_rows": score_rows, "score_record_ids": score_ids,
            "score_cycles": score_cycles, "present": present,
            "champion": champion, "p_pos_original": p_pos, "p_a02": p_a02,
            "entry_friction": entry_friction, "text_load_group": text_load_group,
            "coverage": coverage,
        })
        require(np.array_equal(origin_record["score_rows"], score_rows)
                and np.array_equal(origin_record["score_record_ids"], score_ids)
                and np.array_equal(origin_record["present"], present),
                f"E{origin} predictor outputs do not share the exact query row map")
        model_key = architecture.lower()
        origin_record[f"aux_{model_key}"] = full_aux
        origin_record[model_key] = candidate
        fit_receipt_hashes[fit_id] = sha256_file(receipt_path)
        fit_artifact_hashes[fit_id] = sha256_file(model_path)
        del model, score_counts, score_structured, replay
        gc.collect()

    # E67 and E75 share late query IDs; baseline/route values must be identical.
    e67, e75 = data_by_origin[67], data_by_origin[75]
    late_idx_67 = np.flatnonzero((e67["score_cycles"] >= 76) & (e67["score_cycles"] <= 79))
    id_to_67 = {record_id: i for i, record_id in enumerate(e67["score_record_ids"])}
    for i, record_id in enumerate(e75["score_record_ids"]):
        j = id_to_67.get(record_id)
        require(j is not None and e67["score_cycles"][j] == e75["score_cycles"][i],
                "E75 query IDs/cycles do not map to E67's frozen late query")
        require(np.array_equal(np.asarray([e67["champion"][j]]).view(np.uint64),
                               np.asarray([e75["champion"][i]]).view(np.uint64)),
                "E67/E75 champion probabilities differ on overlapping query IDs")
        require(e67["present"][j] == e75["present"][i],
                "E67/E75 availability differs on overlapping query IDs")

    payload: dict[str, Any] = {}
    prelabel_origins: dict[str, Any] = {}
    for origin, record in data_by_origin.items():
        prefix = f"E{origin}_"
        for key, value in record.items():
            payload[prefix + key] = value
        prelabel_origins[str(origin)] = {
            "rows": len(record["score_rows"]),
            "record_ids_sha256": sha256_array(record["score_record_ids"]),
            "row_positions_sha256": sha256_array(record["score_rows"]),
            "cycles_sha256": sha256_array(record["score_cycles"]),
            "present_rows": int(record["present"].sum()),
            "missing_rows": int((~record["present"]).sum()),
            "missing_route_exact_for_dan": bool(np.array_equal(
                record["dan"][~record["present"]].view(np.uint64),
                record["champion"][~record["present"]].view(np.uint64))),
            "missing_route_exact_for_linear": bool(np.array_equal(
                record["linear"][~record["present"]].view(np.uint64),
                record["champion"][~record["present"]].view(np.uint64))),
        }
    payload["prelabel_metadata_json"] = np.asarray(json.dumps({
        "candidate_id": "CLS_DAN_V1", "freeze_signature": freeze["freeze_signature"],
        "fit_receipt_hashes": fit_receipt_hashes, "fit_artifact_hashes": fit_artifact_hashes,
        "evaluation_query_targets_read": False, "test_opened": False,
        "sample_submission_opened": False, "kaggle_contacted": False,
    }, sort_keys=True, ensure_ascii=False, allow_nan=False))
    atomic_npz(PRELABEL_PATH, **payload)
    ready_record = {
        "candidate_id": "CLS_DAN_V1", "status": "PASS_PRELABEL_READY_NO_QUERY_LABELS",
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["source_commit"],
        "reservation_sha256": sha256_file(RESERVATION_PATH),
        "feature_ready_sha256": sha256_file(FEATURES_READY_PATH),
        "fit_receipt_hashes": fit_receipt_hashes,
        "fit_artifact_hashes": fit_artifact_hashes,
        "prelabel_path": PRELABEL_PATH.relative_to(AUDIT).as_posix(),
        "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "origins": prelabel_origins,
        "unique_eval_query_rows": int(len(e67["score_rows"])),
        "primary_unique_expected_rows": 20903,
        "evaluation_query_targets_read": False,
        "metrics_computed": False,
        "test_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    require(len(e67["score_rows"]) == ready_record["primary_unique_expected_rows"],
            "E67 query union no longer contains 20,903 unique evaluation rows")
    atomic_json(PREDICTIONS_READY_PATH, ready_record, write_once=True)
    return ready_record


def arm_evaluation(official_root: Path) -> dict[str, Any]:
    """Record the exact target row set; does not open target values."""
    freeze, _, _ = verify_root_prefit_and_reservation(official_root)
    ready, _ = verify_features_ready(official_root)
    pred_ready = _require_committed_json(PREDICTIONS_READY_PATH)
    _require_committed_file(PRELABEL_PATH)
    require(pred_ready.get("status") == "PASS_PRELABEL_READY_NO_QUERY_LABELS"
            and pred_ready.get("freeze_signature") == freeze["freeze_signature"]
            and pred_ready.get("evaluation_query_targets_read") is False
            and pred_ready.get("metrics_computed") is False
            and pred_ready.get("prelabel_sha256") == sha256_file(PRELABEL_PATH),
            "Committed prelabel state is not eligible for one query-label evaluation")
    require(not (RUN_ROOT / "EVALUATION_ARM.json").exists()
            and not (RUN_ROOT / "EVALUATION_STARTED.json").exists()
            and not (RUN_ROOT / "EVALUATION_CONSUMED.json").exists()
            and not RESULTS_PATH.exists(), "Evaluation boundary files already exist")
    with np.load(PRELABEL_PATH, allow_pickle=False) as archive:
        rows = np.asarray(archive["E67_score_rows"], dtype=np.int64)
        ids = np.asarray(archive["E67_score_record_ids"]).astype(str)
        cycles = np.asarray(archive["E67_score_cycles"], dtype=np.int64)
    arm = {
        "candidate_id": "CLS_DAN_V1", "status": "READY_TO_READ_FROZEN_QUERY_TARGETS",
        "freeze_signature": freeze["freeze_signature"],
        "prediction_ready_sha256": sha256_file(PREDICTIONS_READY_PATH),
        "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "query_rows": len(rows), "query_row_positions_sha256": sha256_array(rows),
        "query_record_ids_sha256": sha256_array(ids), "query_cycles_sha256": sha256_array(cycles),
        "query_target_reader": "CSV row-aware exact record_id/artifact_signal selection",
        "evaluation_target_values_read": False, "metrics_computed": False,
        "max_query_evaluation_attempts": 1,
        "independent_verifier_uses_same_in_memory_labels": True,
        "test_opened": False, "sample_submission_opened": False, "kaggle_contacted": False,
    }
    require(len(rows) == 20903 and np.all((cycles >= 68) & (cycles <= 79)),
            "Evaluation target rows differ from the frozen E67 query union")
    atomic_json(RUN_ROOT / "EVALUATION_ARM.json", arm, write_once=True)
    return arm


def begin_evaluation(official_root: Path) -> dict[str, Any]:
    """Commit-required one-shot scoring marker, before any query target read."""
    freeze, _, _ = verify_root_prefit_and_reservation(official_root)
    arm = _require_committed_json(RUN_ROOT / "EVALUATION_ARM.json")
    _require_committed_file(PRELABEL_PATH)
    require(arm.get("status") == "READY_TO_READ_FROZEN_QUERY_TARGETS"
            and arm.get("freeze_signature") == freeze["freeze_signature"]
            and arm.get("evaluation_target_values_read") is False,
            "Committed evaluation arm is invalid")
    require(not (RUN_ROOT / "EVALUATION_STARTED.json").exists()
            and not (RUN_ROOT / "EVALUATION_CONSUMED.json").exists()
            and not RESULTS_PATH.exists(), "Evaluation is strictly single-attempt")
    marker = {
        "candidate_id": "CLS_DAN_V1", "status": "ONE_SHOT_EVALUATION_STARTED",
        "freeze_signature": freeze["freeze_signature"],
        "evaluation_arm_sha256": sha256_file(RUN_ROOT / "EVALUATION_ARM.json"),
        "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "source_commit": freeze["source_commit"],
        "one_query_label_read_attempt": True,
        "no_retry_after_attempt_marker": True,
        "evaluation_target_values_read": False,
    }
    atomic_json(RUN_ROOT / "EVALUATION_STARTED.json", marker, write_once=True)
    return marker


def _metrics(y: np.ndarray, probability: np.ndarray, present: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    is_present = np.asarray(present, dtype=bool).reshape(-1)
    require(len(target) == len(p) == len(is_present) and len(target) > 0
            and np.isin(target, (0, 1)).all() and np.isfinite(p).all()
            and np.all((p >= 0.0) & (p <= 1.0)), "Evaluation metric arrays are invalid")
    threshold = np.where(is_present, 0.50, 0.53230976129477)
    predicted = (p >= threshold).astype(np.int8)
    return {
        "n": int(len(target)), "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "predicted_positive": int(predicted.sum()),
        "confusion_matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
        "macro_f1": float(f1_score(target, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "precision": float(precision_score(target, predicted, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(target, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, p)) if np.unique(target).size == 2 else None,
        "average_precision": float(average_precision_score(target, p)) if np.any(target == 1) else 0.0,
        "log_loss": float(log_loss(target, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, p)),
    }


def _view_part(record: Mapping[str, Any], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "record_ids": np.asarray(record["score_record_ids"])[mask].astype(str),
        "rows": np.asarray(record["score_rows"], dtype=np.int64)[mask],
        "cycles": np.asarray(record["score_cycles"], dtype=np.int64)[mask],
        "present": np.asarray(record["present"], dtype=bool)[mask],
        "champion": np.asarray(record["champion"], dtype=np.float64)[mask],
        "dan": np.asarray(record["dan"], dtype=np.float64)[mask],
        "linear": np.asarray(record["linear"], dtype=np.float64)[mask],
        "entry_friction": np.asarray(record["entry_friction"]).astype(str)[mask],
        "text_load_group": np.asarray(record["text_load_group"]).astype(str)[mask],
        "coverage": np.asarray(record["coverage"], dtype=np.float64)[mask],
    }


def _concatenate_views(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, np.ndarray]:
    output = {key: np.concatenate((np.asarray(first[key]), np.asarray(second[key])))
              for key in first}
    require(len(np.unique(output["record_ids"])) == len(output["record_ids"]),
            "Primary unique view contains duplicate record IDs")
    return output


def _build_views(data_by_origin: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    e67, e75 = data_by_origin[67], data_by_origin[75]
    c67 = np.asarray(e67["score_cycles"], dtype=np.int64)
    c75 = np.asarray(e75["score_cycles"], dtype=np.int64)
    views = {
        "E67_W4": _view_part(e67, (c67 >= 68) & (c67 <= 71)),
        "E67_W5": _view_part(e67, (c67 >= 72) & (c67 <= 75)),
        "E67_W6": _view_part(e67, (c67 >= 76) & (c67 <= 79)),
        "E67_H12": _view_part(e67, (c67 >= 68) & (c67 <= 79)),
        "E75_W6": _view_part(e75, (c75 >= 76) & (c75 <= 79)),
    }
    early = _view_part(e67, (c67 >= 68) & (c67 <= 75))
    late = _view_part(e75, (c75 >= 76) & (c75 <= 79))
    views["PRIMARY_UNIQUE"] = _concatenate_views(early, late)
    require(len(views["PRIMARY_UNIQUE"]["record_ids"]) == 20903,
            "Primary unique view differs from 20,903 frozen IDs")
    return views


def _subgroup_results(view_name: str, view: Mapping[str, Any], y: np.ndarray,
                      minimum_rows: int = 200, minimum_per_class: int = 30) -> list[dict[str, Any]]:
    n = len(y)
    variables = {
        "stream_availability": np.where(np.asarray(view["present"], dtype=bool), "present", "missing"),
        "entry_friction": np.asarray(view["entry_friction"]).astype(str),
        "fit_only_text_load_quartile": np.asarray(view["text_load_group"]).astype(str),
    }
    output: list[dict[str, Any]] = []
    for variable, values in variables.items():
        require(len(values) == n, f"{view_name} subgroup {variable} does not align")
        for category in sorted(np.unique(values).tolist()):
            mask = values == category
            n_pos = int(np.sum(y[mask] == 1))
            n_neg = int(np.sum(y[mask] == 0))
            support = {"n": int(mask.sum()), "positive": n_pos, "negative": n_neg,
                       "minimum_rows": minimum_rows, "minimum_per_class": minimum_per_class}
            if support["n"] < minimum_rows or n_pos < minimum_per_class or n_neg < minimum_per_class:
                output.append({"view": view_name, "variable": variable, "group": category,
                               "support": support, "status": "UNSUPPORTED"})
                continue
            group_y = y[mask]
            metrics = {model: _metrics(group_y, np.asarray(view[model])[mask],
                                       np.asarray(view["present"], dtype=bool)[mask])
                       for model in ("champion", "dan", "linear")}
            output.append({
                "view": view_name, "variable": variable, "group": category,
                "support": support, "status": "SUPPORTED", "metrics": metrics,
                "dan_macro_f1_harm_vs_champion": metrics["champion"]["macro_f1"] - metrics["dan"]["macro_f1"],
                "dan_positive_recall_harm_vs_champion": metrics["champion"]["positive_recall"] - metrics["dan"]["positive_recall"],
                "linear_macro_f1_harm_vs_champion": metrics["champion"]["macro_f1"] - metrics["linear"]["macro_f1"],
                "linear_positive_recall_harm_vs_champion": metrics["champion"]["positive_recall"] - metrics["linear"]["positive_recall"],
            })
    return output


def evaluate_once(official_root: Path) -> dict[str, Any]:
    """Read the committed query target rows once and score all views in memory."""
    freeze, _, _ = verify_root_prefit_and_reservation(official_root)
    features_ready = _require_committed_json(FEATURES_READY_PATH)
    arm = _require_committed_json(RUN_ROOT / "EVALUATION_ARM.json")
    started_marker = _require_committed_json(RUN_ROOT / "EVALUATION_STARTED.json")
    pred_ready = _require_committed_json(PREDICTIONS_READY_PATH)
    _require_committed_file(PRELABEL_PATH)
    require(arm.get("status") == "READY_TO_READ_FROZEN_QUERY_TARGETS"
            and started_marker.get("status") == "ONE_SHOT_EVALUATION_STARTED"
            and arm.get("freeze_signature") == freeze["freeze_signature"]
            and started_marker.get("evaluation_arm_sha256") == sha256_file(RUN_ROOT / "EVALUATION_ARM.json")
            and started_marker.get("prelabel_sha256") == sha256_file(PRELABEL_PATH)
            and pred_ready.get("evaluation_query_targets_read") is False,
            "Committed one-shot evaluation boundary is invalid")
    require(not (RUN_ROOT / "EVALUATION_CONSUMED.json").exists() and not RESULTS_PATH.exists(),
            "Query evaluation attempt has already been consumed; no rerun")

    consumed = {
        "candidate_id": "CLS_DAN_V1", "status": "EVALUATION_ATTEMPT_CONSUMED",
        "freeze_signature": freeze["freeze_signature"],
        "evaluation_start_sha256": sha256_file(RUN_ROOT / "EVALUATION_STARTED.json"),
        "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "query_read_status": "READ_PENDING_AT_MARKER_CREATION",
        "no_retry_after_marker": True,
    }
    atomic_json(RUN_ROOT / "EVALUATION_CONSUMED.json", consumed, write_once=True)
    with np.load(PRELABEL_PATH, allow_pickle=False) as archive:
        data_by_origin: dict[int, dict[str, Any]] = {}
        for origin in ORIGINS:
            prefix = f"E{origin}_"
            data_by_origin[origin] = {
                "score_rows": np.asarray(archive[prefix + "score_rows"], dtype=np.int64).copy(),
                "score_record_ids": np.asarray(archive[prefix + "score_record_ids"]).astype(str),
                "score_cycles": np.asarray(archive[prefix + "score_cycles"], dtype=np.int64).copy(),
                "present": np.asarray(archive[prefix + "present"], dtype=bool).copy(),
                "champion": np.asarray(archive[prefix + "champion"], dtype=np.float64).copy(),
                "dan": np.asarray(archive[prefix + "dan"], dtype=np.float64).copy(),
                "linear": np.asarray(archive[prefix + "linear"], dtype=np.float64).copy(),
                "entry_friction": np.asarray(archive[prefix + "entry_friction"]).astype(str),
                "text_load_group": np.asarray(archive[prefix + "text_load_group"]).astype(str),
                "coverage": np.asarray(archive[prefix + "coverage"], dtype=np.float64).copy(),
            }
    e67 = data_by_origin[67]
    y_e67 = _read_exact_targets(
        Path(official_root) / "data/raw/classification/Train.csv",
        e67["score_rows"], e67["score_record_ids"], purpose="single frozen query evaluation",
    )
    y_by_id = {record_id: int(y) for record_id, y in zip(e67["score_record_ids"], y_e67)}
    require(len(y_by_id) == len(y_e67) == 20903,
            "E67 query labels do not form the expected unique one-read evaluation set")
    views = _build_views(data_by_origin)
    view_labels: dict[str, np.ndarray] = {}
    for name, view in views.items():
        view_labels[name] = np.asarray([y_by_id[record_id] for record_id in view["record_ids"]], dtype=np.int8)
    evaluation_views: dict[str, Any] = {}
    verifier_input: dict[str, dict[str, Any]] = {}
    subgroup_rows: list[dict[str, Any]] = []
    for name, view in views.items():
        labels = view_labels[name]
        model_metrics = {model: _metrics(labels, np.asarray(view[model]), np.asarray(view["present"]))
                         for model in ("champion", "dan", "linear")}
        evaluation_views[name] = {
            "n": len(labels), "cycle_min": int(np.min(view["cycles"])),
            "cycle_max": int(np.max(view["cycles"])),
            "metrics": model_metrics,
            "dan_gain_vs_champion": model_metrics["dan"]["macro_f1"] - model_metrics["champion"]["macro_f1"],
            "linear_gain_vs_champion": model_metrics["linear"]["macro_f1"] - model_metrics["champion"]["macro_f1"],
            "dan_minus_linear": model_metrics["dan"]["macro_f1"] - model_metrics["linear"]["macro_f1"],
            "mean_token_coverage": float(np.mean(view["coverage"])) if len(view["coverage"]) else None,
        }
        for model in ("champion", "dan", "linear"):
            verifier_input[f"{name}::{model}"] = {
                "probability": np.asarray(view[model], dtype=np.float64),
                "present": np.asarray(view["present"], dtype=bool),
            }
        subgroup_rows.extend(_subgroup_results(name, view, labels))

    verifier_by_view: dict[str, Any] = {}
    for name, view in views.items():
        local = {model: verifier_input[f"{name}::{model}"] for model in ("champion", "dan", "linear")}
        verifier_by_view[name] = verify_metric_map(view_labels[name], local)
    verification_pass = all(row["status"] == "PASS" for row in verifier_by_view.values())

    per_cycle = []
    for origin, record in data_by_origin.items():
        cycles = np.asarray(record["score_cycles"], dtype=np.int64)
        for cycle in sorted(np.unique(cycles).tolist()):
            mask = cycles == cycle
            ids = np.asarray(record["score_record_ids"]).astype(str)[mask]
            labels = np.asarray([y_by_id[value] for value in ids], dtype=np.int8)
            metric_c = _metrics(labels, np.asarray(record["champion"])[mask], np.asarray(record["present"])[mask])
            metric_d = _metrics(labels, np.asarray(record["dan"])[mask], np.asarray(record["present"])[mask])
            metric_l = _metrics(labels, np.asarray(record["linear"])[mask], np.asarray(record["present"])[mask])
            per_cycle.append({
                "origin": origin, "cycle": int(cycle), "n": int(mask.sum()),
                "dan_gain_vs_champion": metric_d["macro_f1"] - metric_c["macro_f1"],
                "dan_direction": "POSITIVE" if metric_d["macro_f1"] > metric_c["macro_f1"] else
                                 "NEGATIVE" if metric_d["macro_f1"] < metric_c["macro_f1"] else "TIE",
                "linear_gain_vs_champion": metric_l["macro_f1"] - metric_c["macro_f1"],
                "linear_direction": "POSITIVE" if metric_l["macro_f1"] > metric_c["macro_f1"] else
                                    "NEGATIVE" if metric_l["macro_f1"] < metric_c["macro_f1"] else "TIE",
            })

    required_views = ("PRIMARY_UNIQUE", "E67_H12", "E75_W6")
    primary_gain = evaluation_views["PRIMARY_UNIQUE"]["dan_gain_vs_champion"]
    e67_gain = evaluation_views["E67_H12"]["dan_gain_vs_champion"]
    e75_gain = evaluation_views["E75_W6"]["dan_gain_vs_champion"]
    primary_matched = evaluation_views["PRIMARY_UNIQUE"]["dan_minus_linear"]
    e75_control_gain = evaluation_views["E75_W6"]["linear_gain_vs_champion"]
    supported_subgroups = [row for row in subgroup_rows if row["status"] == "SUPPORTED"]
    each_view_supported = all(any(row["view"] == view and row["status"] == "SUPPORTED"
                                  for row in subgroup_rows) for view in views)
    subgroup_science = each_view_supported and all(
        row["dan_macro_f1_harm_vs_champion"] <= 0.01
        and row["dan_positive_recall_harm_vs_champion"] <= 0.02
        for row in supported_subgroups
    )
    subgroup_exploratory = each_view_supported and all(
        row["dan_macro_f1_harm_vs_champion"] <= 0.02
        and row["dan_positive_recall_harm_vs_champion"] <= 0.03
        for row in supported_subgroups
    )
    # Exact missing-route booleans are included in the immutable readiness receipt.
    route_check = all(
        pred_ready["origins"][str(origin)]["missing_route_exact_for_dan"]
        and pred_ready["origins"][str(origin)]["missing_route_exact_for_linear"]
        for origin in ORIGINS
    )
    unique_id_check = len(np.unique(views["PRIMARY_UNIQUE"]["record_ids"])) == 20903
    probability_check = all(np.isfinite(np.asarray(views[name][model])).all()
                            and np.all((np.asarray(views[name][model]) >= 0.0)
                                       & (np.asarray(views[name][model]) <= 1.0))
                            for name in views for model in ("champion", "dan", "linear"))
    qa_pass = route_check and unique_id_check and probability_check and verification_pass
    scientific_components = {
        "primary_gain_vs_champion_at_least_0_0005": primary_gain >= 0.0005,
        "E67_H12_gain_nonnegative": e67_gain >= 0.0,
        "E75_W6_gain_nonnegative": e75_gain >= 0.0,
        "primary_DAN_minus_LINEAR_at_least_0_0002": primary_matched >= 0.0002,
        "E75_LINEAR_control_gain_nonnegative": e75_control_gain >= 0.0,
        "supported_all_view_subgroups_within_0_01_f1_0_02_recall": subgroup_science,
        "identity_route_probability_and_metric_QA": qa_pass,
    }
    scientific_pass = all(scientific_components.values())
    exploratory_performance = ((primary_gain > 0.0 and e75_gain >= -0.0005)
                               or (e75_gain >= 0.0005 and primary_gain >= -0.0005))
    exploratory_components = {
        "campaign_primary_or_E75_performance_rule": exploratory_performance,
        "supported_all_view_subgroups_within_0_02_f1_0_03_recall": subgroup_exploratory,
        "identity_route_probability_and_metric_QA": qa_pass,
    }
    exploratory_pass = all(exploratory_components.values())
    mechanism_pass = primary_matched >= 0.0002

    fit_receipt_records = {fit_id: read_json(RUN_ROOT / "units" / "receipts" / f"{fit_id}.json")
                           for fit_id in FIT_IDS}
    fit_receipts = {fit_id: sha256_file(RUN_ROOT / "units" / "receipts" / f"{fit_id}.json")
                    for fit_id in FIT_IDS}
    prefix_label_reads_by_fit = {fit_id: int(record["fit_target_values_read"])
                                 for fit_id, record in fit_receipt_records.items()}
    e75_overlap_rows = int(features_ready["origins"]["75"][
        "fit_label_overlap_with_E67_query_cycles_68_75_rows"]
    )
    e75_overlap_prefix_read_occurrences = e75_overlap_rows * sum(
        1 for fit_id, (origin, _) in FIT_IDS.items() if origin == 75
    )
    result = {
        "candidate_id": "CLS_DAN_V1",
        "status": "PASS_EVALUATED" if verification_pass and qa_pass else "FAIL_EVALUATION_QA",
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["source_commit"],
        "evaluation_started_sha256": sha256_file(RUN_ROOT / "EVALUATION_STARTED.json"),
        "evaluation_attempt_consumed_sha256": sha256_file(RUN_ROOT / "EVALUATION_CONSUMED.json"),
        "prelabel_sha256": sha256_file(PRELABEL_PATH),
        "fit_receipt_hashes": fit_receipts,
        "query_target_read": {
            "status": "READ_ONCE_FOR_FROZEN_E67_UNION",
            "evaluation_rows": len(y_e67), "evaluation_label_values_read_once": len(y_e67),
            "positive": int(y_e67.sum()), "negative": int(len(y_e67) - y_e67.sum()),
            "row_positions_sha256": sha256_array(e67["score_rows"]),
            "record_ids_sha256": sha256_array(e67["score_record_ids"]),
            "label_vector_sha256": sha256_array(y_e67),
            "prefix_fit_label_value_reads_by_fit": prefix_label_reads_by_fit,
            "prefix_fit_label_value_reads_total": int(sum(prefix_label_reads_by_fit.values())),
            "E75_fit_prefix_rows_overlapping_E67_query_cycles_68_75": e75_overlap_rows,
            "overlap_prefix_label_read_occurrences_in_E75_fits": e75_overlap_prefix_read_occurrences,
            "overlap_note": "E75 cycles 68-75 targets are read for its own two prefix fits before the later E67 evaluation; primary unique predictions for those rows come from E67.",
            "independent_metric_verifier_reused_same_in_memory_labels": True,
            "test_opened": False, "sample_submission_opened": False, "kaggle_contacted": False,
        },
        "primary_unique_construction": "E67 cycles 68-75 plus E75 cycles 76-79",
        "views": evaluation_views,
        "subgroups": subgroup_rows,
        "per_cycle_signs": per_cycle,
        "independent_verification": verifier_by_view,
        "gates": {
            "scientific": {"status": "PASS" if scientific_pass else "FAIL",
                           "components": scientific_components},
            "exploratory_eligibility": {"status": "PASS" if exploratory_pass else "FAIL",
                                         "components": exploratory_components},
            "mechanism_contrast": {
                "status": "PASS" if mechanism_pass else "FAIL",
                "definition": "predeclared primary DAN-minus-LINEAR Macro F1 >= +0.0002",
                "primary_DAN_minus_LINEAR": primary_matched,
                "threshold": 0.0002,
            },
        },
        "qa": {"route_exact": route_check, "unique_primary_ids": unique_id_check,
               "finite_probabilities": probability_check, "independent_metrics_match": verification_pass,
               "all_views_have_supported_subgroup": each_view_supported,
               "fit_ids": list(FIT_IDS), "fit_cap": 4, "retry_count": 0,
               "query_labels_read_once": True},
        "interpretation_limits": [
            "E67/E75 temporal views overlap and are development evidence, not independent fresh holdouts.",
            "The LINEAR control shares the learned E32 map but not DAN capacity/regularization paths; contrast is not a causal isolation of nonlinearity.",
            "The learned unigram map is a reparameterized linear projection; it does not establish token semantics.",
            "Missing-stream rows use the exact champion fallback and are excluded from auxiliary fitting.",
            "These internal validation results do not authorize full-Train refit, Test inference, or Kaggle submission.",
        ],
    }
    atomic_json(RESULTS_PATH, result, write_once=True)
    atomic_json(RUN_ROOT / "INDEPENDENT_VERIFICATION.json", {
        "candidate_id": "CLS_DAN_V1", "status": "PASS" if verification_pass else "FAIL",
        "label_source": "same one-read in-memory query vector",
        "views": verifier_by_view,
    }, write_once=True)
    _write_results_markdown(result, REPORT_ROOT / "RESULTS.md")
    return result


def _write_results_markdown(result: Mapping[str, Any], path: Path) -> None:
    views = result["views"]
    lines = [
        "# CLS_DAN_V1 results",
        "",
        "Classification pilot result; no Test, SampleSubmission, or Kaggle access.",
        "",
        "## Main metrics",
        "",
        "| View | Rows | Champion Macro F1 | DAN Macro F1 | DAN gain | LINEAR Macro F1 | DAN − LINEAR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("PRIMARY_UNIQUE", "E67_W4", "E67_W5", "E67_W6", "E67_H12", "E75_W6"):
        row = views[name]
        m = row["metrics"]
        lines.append(f"| {name} | {row['n']} | {m['champion']['macro_f1']:.6f} | "
                     f"{m['dan']['macro_f1']:.6f} | {row['dan_gain_vs_champion']:+.6f} | "
                     f"{m['linear']['macro_f1']:.6f} | {row['dan_minus_linear']:+.6f} |")
    lines.extend(["", "## Gate status", ""])
    for name, value in result["gates"].items():
        lines.append(f"- **{name}: {value['status']}**")
    lines.extend(["", "## Interpretation", "",
                  "The table compares fixed 25% auxiliary blends with the saved champion. "
                  "The DAN–LINEAR difference is the predeclared representation/architecture contrast; "
                  "it does not establish a causal effect of nonlinearity or token semantics.", "",
                  "E67 W4/W5/W6/H12 and E75 W6 results are all reported in `RESULTS.json`, "
                  "including subgroup cells, precision/recall, probability metrics, per-cycle signs, "
                  "unsupported cells, and negative findings.", "",
                  "The E67 and E75 windows overlap in development use, and E75 training includes "
                  "cycles 68–75. The primary unique view uses E67 for cycles 68–75 and E75 for "
                  "cycles 76–79. This is not an independent fresh holdout.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen CLS_DAN_V1 pilot runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic", help="Run synthetic-only lifecycle QA")
    synthetic.add_argument("--output-dir", type=Path, required=True)
    for name, help_text in (
        ("preflight", "Read-only source, vocabulary, and matrix lineage verification"),
        ("freeze", "Write the target-free source/input freeze after source commit"),
        ("verify-freeze", "Recheck the committed source/input freeze"),
        ("prepare", "Build frozen-vocabulary feature caches; labels are not read"),
        ("predictions", "Read back four saved models and commit a prelabel bundle"),
        ("arm-evaluation", "Freeze the one query-target row set without reading labels"),
        ("begin-evaluation", "Write the one-shot evaluation marker before label access"),
        ("evaluate", "Read the frozen query labels once and score all views"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--official-root", type=Path, required=True)
    fit = subparsers.add_parser("fit-one", help="Run one reserved official fit; no retries")
    fit.add_argument("--official-root", type=Path, required=True)
    fit.add_argument("--fit-id", choices=tuple(FIT_IDS), required=True)

    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = run_synthetic_lifecycle(args.output_dir)
    elif args.command == "preflight":
        bundle, _, _ = _read_input_bundle(args.official_root)
        result = {
            "candidate_id": "CLS_DAN_V1", "status": "PASS_READ_ONLY_PREFIT",
            "official_train_sha256": EXPECTED_TRAIN_SHA,
            "official_root": bundle["official_root"],
            "official_head": bundle["official_head"],
            "origins": bundle["origins"],
            "input_file_count": len(bundle["input_file_hashes"]),
            "source_hash_count": len(bundle["local_source_hashes"]),
            "query_labels_read": False, "fit_labels_read": False,
            "test_opened": False, "sample_submission_opened": False,
        }
    elif args.command == "freeze":
        result = build_source_input_freeze(args.official_root)
    elif args.command == "verify-freeze":
        result = verify_frozen_sources(args.official_root)
    elif args.command == "prepare":
        result = prepare_features(args.official_root)
    elif args.command == "fit-one":
        result = fit_one(args.official_root, args.fit_id)
    elif args.command == "predictions":
        result = build_prelabel_predictions(args.official_root)
    elif args.command == "arm-evaluation":
        result = arm_evaluation(args.official_root)
    elif args.command == "begin-evaluation":
        result = begin_evaluation(args.official_root)
    elif args.command == "evaluate":
        result = evaluate_once(args.official_root)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
