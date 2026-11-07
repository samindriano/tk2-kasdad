"""Run the frozen CLS_PPMI_CONTEXT_V1 pilot without opening query labels early."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

CORE_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
from cls_ppmi_context_core_v1 import (  # noqa: E402
    EMBEDDING_DIM,
    SHUFFLE_SEED,
    assert_prefix_cycles,
    binary_metrics,
    canonical_sha256,
    cooccurrence_counts,
    document_means,
    ordered_unigram_pairs,
    ordered_unigrams,
    positive_ppmi,
    release_advisory_byte_lock,
    resource_bytes_for_csr,
    sha256_array,
    shuffle_documents,
    tokenize_streams,
    try_advisory_byte_lock,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/optimization_v3/CLS_PPMI_CONTEXT_V1"
EXPERIMENT_ROOT = ROOT / "experiments/CLS_PPMI_CONTEXT_V1"
RUNTIME_ROOT = EXPERIMENT_ROOT / "runtime"
FREEZE_PATH = REPORT_ROOT / "SOURCE_FREEZE.json"
SYNTHETIC_PATH = REPORT_ROOT / "SYNTHETIC_QA.json"
FIT_LEDGER_PATH = EXPERIMENT_ROOT / "fit_ledger.jsonl"
PREFLIGHT_PATH = EXPERIMENT_ROOT / "PREFLIGHT.json"
FIT_SUMMARY_PATH = EXPERIMENT_ROOT / "FIT_SUMMARY.json"
PRELABEL_PATH = EXPERIMENT_ROOT / "PRELABEL_READY.json"
RESULTS_PATH = EXPERIMENT_ROOT / "RESULTS.json"
REPORT_PATH = EXPERIMENT_ROOT / "REPORT.md"
PRELABEL_VERIFICATION_PATH = EXPERIMENT_ROOT / "PRELABEL_VERIFICATION.json"
POSTLABEL_VERIFICATION_PATH = EXPERIMENT_ROOT / "POSTLABEL_VERIFICATION.json"

DEFAULT_SOURCE_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2")
DEFAULT_AUDIT_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-simplification-audit-v1")
TRAIN_REL = Path("data/raw/classification/Train.csv")
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
ORIGINS = (67, 75)
TEXT_WIDTH = 20_000
STRUCTURED_WIDTHS = {67: 284, 75: 370}
MISSING_THRESHOLD = 0.53230976129477
CHAMPION_THRESHOLD = 0.50
MIN_START_AVAILABLE = 1024**3
MIN_RESERVE_AVAILABLE = 512 * 1024**2
MAX_THREADS = 2
FIT_ORDER = ((67, "REAL"), (67, "SHUFFLED"), (75, "REAL"), (75, "SHUFFLED"))
HGB_PARAMS = {
    "loss": "log_loss", "learning_rate": 0.05, "max_iter": 200,
    "max_leaf_nodes": 15, "min_samples_leaf": 30,
    "l2_regularization": 1.0, "max_bins": 255,
    "early_stopping": False, "random_state": 42,
}
OWNED_SOURCE_PATHS = (
    "src/models/cls_ppmi_context_core_v1.py",
    "scripts/cls_ppmi_context_v1.py",
    "scripts/verify_cls_ppmi_context_v1.py",
    "tests/test_cls_ppmi_context_v1.py",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/PROTOCOL.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/SYNTHETIC_QA.json",
    "experiments/CLS_PPMI_CONTEXT_V1/.gitignore",
)


class PilotError(RuntimeError):
    pass


class LockBusy(PilotError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, write_once: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_once and path.exists():
        raise PilotError(f"Write-once output already exists: {path}")
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                           indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                   text=True, encoding="utf-8").strip()


def committed_file_bytes(repository: Path, relative: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repository), "show", f"HEAD:{relative}"])


def assert_committed_exact(repository: Path, path: Path, relative: str) -> None:
    try:
        committed = committed_file_bytes(repository, relative)
    except subprocess.CalledProcessError as exc:
        raise PilotError(f"Required file is not committed at HEAD: {relative}") from exc
    disk = path.read_bytes()
    if committed.replace(b"\r\n", b"\n") != disk.replace(b"\r\n", b"\n"):
        raise PilotError(f"Committed and on-disk bytes differ: {relative}")


def _load_reference_helper(audit_root: Path):
    helper_path = audit_root / "scripts/cls_pos_poisson_bag2_v1.py"
    if not helper_path.is_file():
        raise PilotError(f"Authoritative read-only context loader missing: {helper_path}")
    spec = importlib.util.spec_from_file_location("_cls_pos_bag2_readonly", helper_path)
    if spec is None or spec.loader is None:
        raise PilotError("Could not load the read-only source context helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_hashes(source_root: Path, audit_root: Path,
                   helper: Any | None = None,
                   contexts: dict[int, dict[str, Any]] | None = None) -> dict[str, str]:
    helper = helper or _load_reference_helper(audit_root)
    paths: set[Path] = set()
    for relative in (*helper.SOURCE_FILES, *helper.EXTERNAL_SOURCE_FILES, *helper.BASE_INPUT_FILES):
        paths.add((audit_root if relative in helper.SOURCE_FILES else source_root) / relative)
    paths.add(audit_root / "reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_CAMPAIGN.md")
    paths.add(audit_root / "reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_VOCAB_PARITY.json")
    for origin in ORIGINS:
        suffix = f"E{origin}"
        manifest_path = source_root / helper.PILOT_ROOT_REL / f"MATRIX_MANIFEST_{suffix}.json"
        paths.add(manifest_path)
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            for item in manifest.get("matrices", {}).values():
                matrix_path = Path(str(item["path"]))
                paths.add(matrix_path if matrix_path.is_absolute() else source_root / matrix_path)
        paths.add(source_root / helper.PILOT_ROOT_REL / suffix / f"predictions_{suffix}.npz")
        a02 = source_root / helper.A02_RUN_REL / suffix / "observed"
        for name in ("model.joblib", "fit_receipt.json", "predictions.json", "predictions.npz"):
            paths.add(a02 / name)
        pos_id = f"POS_BAG_OFFSET_V1_{suffix}_REAL"
        paths.add(source_root / helper.POS_RUN_REL / "units" / f"{pos_id}.npz")
        paths.add(source_root / helper.POS_RUN_REL / "receipts" / f"{pos_id}.json")
        paths.add(source_root / helper.POS_RUN_REL / suffix / f"predictions_{suffix}.npz")
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_{suffix}_BAG_LR"
        paths.add(source_root / helper.PILOT_ROOT_REL / "units" / f"{bag_id}.joblib")
        paths.add(source_root / helper.PILOT_ROOT_REL / "receipts" / f"{bag_id}.json")
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise PilotError("Frozen source files are missing: " + "; ".join(missing[:8]))
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths, key=str)}


def _owned_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in OWNED_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise PilotError(f"Owned frozen source is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _sparse_identical(left: sparse.spmatrix, right: sparse.spmatrix) -> bool:
    a = sparse.csr_matrix(left, dtype=np.float64)
    b = sparse.csr_matrix(right, dtype=np.float64)
    if a.shape != b.shape:
        return False
    difference = a - b
    difference.eliminate_zeros()
    return difference.nnz == 0


def _check_vocabulary_and_c2b(source_root: Path, audit_root: Path,
                              helper: Any, context: dict[str, Any],
                              feature_frame: pd.DataFrame,
                              root_parity: dict[str, Any]) -> dict[str, Any]:
    origin = int(context["origin"])
    suffix = f"E{origin}"
    model_path = source_root / helper.A02_RUN_REL / suffix / "observed/model.joblib"
    receipt_path = source_root / helper.A02_RUN_REL / suffix / "observed/fit_receipt.json"
    model = joblib.load(model_path)
    vectorizer = model.text_space.vectorizer
    terms = ordered_unigrams(vectorizer)
    unigram_pairs = ordered_unigram_pairs(vectorizer)
    parity = root_parity["origins"][str(origin)]
    if sha256_file(model_path) != parity["model_sha256"]:
        raise PilotError(f"A02 model hash differs from the root parity receipt for {suffix}")
    if sha256_file(receipt_path) != parity["fit_receipt_sha256"]:
        raise PilotError(f"A02 fit receipt hash differs from the root parity receipt for {suffix}")
    if context["manifest_sha256"] != parity["manifest_sha256"]:
        raise PilotError(f"POS matrix manifest hash differs from root parity receipt for {suffix}")
    if len(vectorizer.vocabulary_) != TEXT_WIDTH or len(terms) != int(parity["unigram_count"]):
        raise PilotError(f"Frozen TF-IDF/unigram width differs from root parity for {suffix}")
    unigram_order_sha = canonical_sha256(unigram_pairs)
    if unigram_order_sha != parity["unigram_order_sha256"]:
        raise PilotError(f"Frozen ordered unigram map differs from root parity for {suffix}")
    ordered_vocabulary = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    vocabulary_sha = canonical_sha256([(str(term), int(index))
                                       for term, index in ordered_vocabulary])
    if vocabulary_sha != model.text_space.vocabulary_sha256:
        raise PilotError(f"Loaded A02 vocabulary differs from its saved text-space receipt for {suffix}")
    if context["manifest"]["shared_preprocessing_sha256"]["tfidf_vocabulary_idf"] != parity["tfidf_vocab_idf_sha256"]:
        raise PilotError(f"TF-IDF vocabulary/IDF digest differs from POS manifest for {suffix}")
    idf_sha = hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes(order="C")).hexdigest()
    if idf_sha != model.text_space.idf_sha256:
        raise PilotError(f"Loaded A02 vectorizer IDF differs from its saved text-space receipt for {suffix}")
    width = int(context["matrices"]["fit_base"].shape[1] - TEXT_WIDTH - 1)
    if width != STRUCTURED_WIDTHS[origin] or width != int(parity["structured_width"]):
        raise PilotError(f"C2_B structured block width differs from the frozen map for {suffix}")
    fit_frame = feature_frame.iloc[np.asarray(context["fit_rows"], dtype=np.int64)]
    score_frame = feature_frame.iloc[np.asarray(context["score_rows"], dtype=np.int64)]
    fit_text, fit_missing = model.text_space.transform(fit_frame["signal_stream"])
    score_text, score_missing = model.text_space.transform(score_frame["signal_stream"])
    fit_base = sparse.csr_matrix(context["matrices"]["fit_base"], dtype=np.float64)
    score_base = sparse.csr_matrix(context["matrices"]["score_base"], dtype=np.float64)
    text_start = width
    text_stop = text_start + TEXT_WIDTH
    if not _sparse_identical(fit_base[:, text_start:text_stop], fit_text):
        raise PilotError(f"Frozen C2_B text block does not replay A02 vectorizer for {suffix} fit rows")
    if not _sparse_identical(score_base[:, text_start:text_stop], score_text):
        raise PilotError(f"Frozen C2_B text block does not replay A02 vectorizer for {suffix} score rows")
    fit_missing_column = np.asarray(fit_base[:, -1].toarray()).reshape(-1).astype(bool)
    score_missing_column = np.asarray(score_base[:, -1].toarray()).reshape(-1).astype(bool)
    if not np.array_equal(fit_missing_column, fit_missing) or not np.array_equal(score_missing_column, score_missing):
        raise PilotError(f"C2_B final missingness feature does not replay for {suffix}")
    if not np.array_equal(~score_missing, np.asarray(context["present"], dtype=bool)):
        raise PilotError(f"A02 and POS score text availability differs for {suffix}")
    structured_fit = fit_base[:, :width]
    structured_score = score_base[:, :width]
    if not np.isfinite(structured_fit.data).all() or not np.isfinite(structured_score.data).all():
        raise PilotError(f"Frozen C2_B structured block has nonfinite values for {suffix}")
    return {
        "origin": origin, "model_path": str(model_path.resolve()),
        "model_sha256": parity["model_sha256"], "fit_receipt_sha256": parity["fit_receipt_sha256"],
        "manifest_sha256": context["manifest_sha256"],
        "tfidf_vocabulary_idf_sha256": parity["tfidf_vocab_idf_sha256"],
        "vocabulary_size": len(vectorizer.vocabulary_), "ordered_unigram_count": len(terms),
        "ordered_unigram_sha256": unigram_order_sha,
        "ordered_unigram_digest_rule": "canonical JSON of (term, original TF-IDF column index) pairs in index order",
        "structured_width": width, "text_width": TEXT_WIDTH,
        "feature_order": "C2_B structured columns, frozen TF-IDF columns, final missingness flag",
        "fit_text_block_replay_exact": True, "score_text_block_replay_exact": True,
        "missing_flag_replay_exact": True, "score_availability_matches_POS": True,
        "fit_present_streams": int((~fit_missing).sum()),
        "fit_missing_streams": int(fit_missing.sum()),
        "score_present_streams": int((~score_missing).sum()),
        "score_missing_streams": int(score_missing.sum()),
        "vectorizer": vectorizer,
    }


def _run_synthetic() -> dict[str, Any]:
    qa: dict[str, Any] = {"schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
                          "status": "PASS", "official_source_opened": False,
                          "query_targets_read": False, "test_or_sample_opened": False,
                          "cases": {}}
    docs = [[0, 1, 2, -1, 0, 1], [2, 1], [3]]
    actual = cooccurrence_counts(docs, 4)
    expected_rows: list[int] = []
    expected_cols: list[int] = []
    for doc in docs:
        for i, left in enumerate(doc):
            if left < 0:
                continue
            for distance in (1, 2):
                j = i + distance
                if j >= len(doc) or doc[j] < 0 or any(x < 0 for x in doc[i + 1:j]):
                    continue
                right = doc[j]
                expected_rows.extend((left, right))
                expected_cols.extend((right, left))
    expected = sparse.coo_matrix((np.ones(len(expected_rows), dtype=np.int32),
                                  (expected_rows, expected_cols)),
                                 shape=(4, 4), dtype=np.int32).tocsr()
    if not sparse.isspmatrix_csr(actual) or (actual != expected).nnz:
        raise PilotError("Synthetic sparse pair accumulation disagrees with direct enumeration")
    qa["cases"]["pair_enumeration_vs_sparse"] = "PASS"
    if cooccurrence_counts([[0, -1, 1], [2], [3]], 4).nnz != 0:
        raise PilotError("Synthetic OOV/document boundary check failed")
    qa["cases"]["no_oov_or_document_gap_crossing"] = "PASS"
    original = [[1, 2, 1, -1, 3], [4, 4, 2], [], [8]]
    shuffled = shuffle_documents(original, seed=SHUFFLE_SEED)
    if [sorted(row) for row in original] != [sorted(row) for row in shuffled]:
        raise PilotError("Synthetic shuffle changed within-document token marginals")
    qa["cases"]["marginal_permutation_invariance"] = "PASS"
    ppmi = positive_ppmi(sparse.csr_matrix(np.asarray([[0, 3], [1, 0]], dtype=np.int32)))
    if not np.isclose(float(ppmi[0, 1]), np.log(4.0 / 3.0), rtol=0.0, atol=1e-12):
        raise PilotError("Synthetic manual PPMI calculation failed")
    qa["cases"]["manual_ppmi_arithmetic_and_sparse_zeros"] = "PASS"
    sample_counts = cooccurrence_counts([[0, 1, 2, 3], [3, 2, 1, 0], [0, 2, 1, 3]], 4)
    sample_ppmi = positive_ppmi(sample_counts)
    with threadpool_limits(limits=MAX_THREADS):
        first_model = TruncatedSVD(n_components=2, random_state=42, n_iter=5)
        second_model = TruncatedSVD(n_components=2, random_state=42, n_iter=5)
        first = first_model.fit_transform(sample_ppmi)
        second = second_model.fit_transform(sample_ppmi)
    if not np.allclose(first, second, rtol=0.0, atol=1e-12):
        raise PilotError("Synthetic SVD deterministic replay failed")
    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact = Path(temporary_directory) / "term_embeddings.npz"
        np.savez_compressed(artifact, embedding=first)
        with np.load(artifact, allow_pickle=False) as saved:
            restored = saved["embedding"]
        if not np.array_equal(first, restored) or sha256_array(first) != sha256_array(restored):
            raise PilotError("Synthetic SVD artifact readback failed")
        original_hash = sha256_file(artifact)
        artifact.write_bytes(artifact.read_bytes() + b"tamper")
        if sha256_file(artifact) == original_hash:
            raise PilotError("Synthetic artifact tamper detection failed")
    qa["cases"]["svd_repeat_save_reload_and_tamper"] = "PASS"
    means, mean_stats = document_means([[0, 1, -1], [-1, -1], []],
                                       np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                                       missing_mask=[False, False, True])
    if not np.array_equal(means[1], np.zeros(2)) or mean_stats["known_token_occurrences"] != 2:
        raise PilotError("Synthetic OOV document mean check failed")
    qa["cases"]["oov_mean_and_zero_fallback"] = "PASS"
    try:
        assert_prefix_cycles([47, 68], 67)
    except ValueError:
        pass
    else:
        raise PilotError("Synthetic train-only prefix spy did not reject a future row")
    qa["cases"]["train_only_prefix_spy"] = "PASS"
    with tempfile.TemporaryDirectory() as temporary_directory:
        lock_path = Path(temporary_directory) / "synthetic_shared.lock"
        lock_path.write_bytes(b"\0")
        child_code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(CORE_DIR)!r}); "
            "from cls_ppmi_context_core_v1 import try_advisory_byte_lock, release_advisory_byte_lock; "
            "p=Path(sys.argv[1]); h=p.open('r+b'); "
            "ok=try_advisory_byte_lock(h); print('LOCKED' if ok else 'BUSY', flush=True); "
            "sys.stdin.readline() if ok else None; "
            "release_advisory_byte_lock(h) if ok else None; h.close(); print('DONE', flush=True)"
        )
        child = subprocess.Popen([sys.executable, "-c", child_code, str(lock_path)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, bufsize=1)
        first_line = child.stdout.readline().strip() if child.stdout is not None else ""
        if first_line != "LOCKED":
            if child.stdin is not None:
                child.stdin.write("\n")
                child.stdin.flush()
            stdout_tail, stderr_tail = child.communicate(timeout=10)
            raise PilotError(f"Synthetic advisory-lock child failed: {first_line}; {stderr_tail[:300]}; {stdout_tail[:100]}")
        with lock_path.open("r+b") as parent_handle:
            parent_acquired_while_held = try_advisory_byte_lock(parent_handle)
            if parent_acquired_while_held:
                release_advisory_byte_lock(parent_handle)
        if child.stdin is None or child.stdout is None:
            raise PilotError("Synthetic advisory-lock child streams are unavailable")
        child.stdin.write("release\n")
        child.stdin.flush()
        released_line = child.stdout.readline().strip()
        done_line = child.stdout.readline().strip()
        return_code = child.wait(timeout=10)
        with lock_path.open("r+b") as parent_handle:
            parent_acquired_after_release = try_advisory_byte_lock(parent_handle)
            if parent_acquired_after_release:
                release_advisory_byte_lock(parent_handle)
        if parent_acquired_while_held or released_line != "DONE" or done_line != "" \
                or return_code != 0 or not parent_acquired_after_release \
                or lock_path.read_bytes() != b"\0":
            raise PilotError("Synthetic advisory lock held/unheld or marker-integrity check failed")
    qa["cases"]["advisory_lock_held_release_reacquire"] = "PASS"
    metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.8], [0, 1, 0, 1])
    if metrics["confusion_matrix_labels_0_1"] != [[1, 1], [1, 1]] or metrics["f1"] != metrics["independent_f1"]:
        raise PilotError("Synthetic independent metrics check failed")
    qa["cases"]["metric_independent_recompute"] = "PASS"
    required = {"schema_version", "pilot_id", "status", "official_source_opened",
                "query_targets_read", "test_or_sample_opened", "cases"}
    if set(qa) != required or len(qa["cases"]) != 9:
        raise PilotError("Synthetic CLI report contract is incomplete")
    qa["cases"]["cli_report_contract"] = "PASS"
    qa["summary"] = {"case_count": len(qa["cases"]), "metrics_fixture": metrics,
                      "mean_fixture": mean_stats, "pair_fixture_nnz": int(actual.nnz)}
    return qa


def command_synthetic() -> None:
    qa = _run_synthetic()
    atomic_json(SYNTHETIC_PATH, qa, write_once=True)
    print(json.dumps({"status": qa["status"], "path": str(SYNTHETIC_PATH),
                      "case_count": len(qa["cases"]), "sha256": sha256_file(SYNTHETIC_PATH)},
                     sort_keys=True))


def command_freeze(source_root: Path, audit_root: Path) -> None:
    source_root = source_root.resolve()
    audit_root = audit_root.resolve()
    if FREEZE_PATH.exists():
        raise PilotError(f"Write-once source freeze already exists: {FREEZE_PATH}")
    if not SYNTHETIC_PATH.is_file() or read_json(SYNTHETIC_PATH).get("status") != "PASS":
        raise PilotError("Passing synthetic QA is required before source/input freeze")
    if (EXPERIMENT_ROOT / "fit_ledger.jsonl").exists() or RUNTIME_ROOT.exists():
        raise PilotError("Fit/representation artifacts already exist before source freeze")
    if git("rev-parse", "HEAD") != "be345d64f666b146390d682ad79d01d908ac0b88":
        raise PilotError("Expected preserved be345d6 worktree HEAD changed before source freeze")
    helper = _load_reference_helper(audit_root)
    plan_relative = "reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_CAMPAIGN.md"
    parity_relative = "reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_VOCAB_PARITY.json"
    plan_path = audit_root / plan_relative
    parity_path = audit_root / parity_relative
    assert_committed_exact(audit_root, plan_path, plan_relative)
    assert_committed_exact(audit_root, parity_path, parity_relative)
    root_campaign = read_json(parity_path) if parity_path.suffix == ".json" else {}
    if root_campaign.get("status") != "PASS" or root_campaign.get("origins") is None:
        raise PilotError("Root vocabulary parity report is not a passing committed receipt")
    train_path = source_root / TRAIN_REL
    if sha256_file(train_path) != TRAIN_SHA256:
        raise PilotError("Official raw Train.csv SHA-256 differs from the frozen source")
    feature_frame = pd.read_csv(train_path, usecols=["record_id", "cycle_code", "signal_stream"],
                                dtype={"record_id": "string"})
    if len(feature_frame) != 50_608 or feature_frame["record_id"].isna().any() \
            or feature_frame["record_id"].duplicated().any():
        raise PilotError("Official feature-only Train row identity check failed")
    cycles = pd.to_numeric(feature_frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if set(np.unique(cycles)) != set(range(47, 80)):
        raise PilotError("Official feature-only Train cycle span differs from 47-79")
    origins: dict[str, Any] = {}
    for origin in ORIGINS:
        context = helper._load_origin_context(source_root, origin, require_fit_labels=False)
        assert_prefix_cycles(cycles[np.asarray(context["fit_rows"], dtype=np.int64)], origin)
        parity = _check_vocabulary_and_c2b(source_root, audit_root, helper, context,
                                           feature_frame, root_campaign)
        parity.pop("vectorizer", None)
        origins[str(origin)] = {
            **parity,
            "fit_rows": int(len(context["fit_rows"])),
            "score_rows": int(len(context["score_rows"])),
            "fit_row_ids_sha256": helper.sha256_array(context["fit_ids"]),
            "score_row_ids_sha256": helper.sha256_array(context["score_ids"]),
            "score_cycle_min": int(context["cycles"].min()),
            "score_cycle_max": int(context["cycles"].max()),
            "position_feature_width": int(context["x_fit"].shape[1]),
            "score_present": int(context["present"].sum()),
            "score_missing": int((~context["present"]).sum()),
        }
    source_files = _source_hashes(source_root, audit_root, helper)
    body = {
        "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1", "created_at_utc": utc_now(),
        "source_worktree_head": git("rev-parse", "HEAD"),
        "source_root": str(source_root), "audit_root": str(audit_root),
        "authoritative_plan": {"path": str(plan_path),
                               "sha256": sha256_file(plan_path),
                               "authorized_root_commit": "e2770ac1b1043ef4cf3855df833053706ab748eb",
                               "audit_head_at_freeze": subprocess.check_output(
                                   ["git", "-C", str(audit_root), "rev-parse", "HEAD"],
                                   text=True, encoding="utf-8").strip()},
        "root_vocab_parity_report": {"path": str(parity_path), "sha256": sha256_file(parity_path)},
        "official_source": {"path": str(train_path), "sha256": TRAIN_SHA256,
                            "rows": len(feature_frame), "cycle_min": int(cycles.min()),
                            "cycle_max": int(cycles.max()),
                            "columns_read": ["record_id", "cycle_code", "signal_stream"],
                            "target_column_read": False, "query_targets_read": False,
                            "test_or_sample_opened": False},
        "origins": origins,
        "source_files": source_files,
        "owned_source_paths": list(OWNED_SOURCE_PATHS),
        "owned_source_sha256": _owned_source_hashes(),
        "synthetic_qa_sha256": sha256_file(SYNTHETIC_PATH),
        "method": {
            "unigram_rule": "no-space terms sorted by frozen vectorizer vocabulary index",
            "tokenizer": "frozen vectorizer.build_preprocessor() then build_tokenizer()",
            "context_graph": "prefix-present streams; directed symmetric distance 1 and 2, unit weight, no OOV or document gap crossing",
            "shuffle": "within-document token permutation incl OOV sentinel; stable row order; numpy.default_rng(42)",
            "ppmi": "max(log(Mij*N/(rowSum_i*colSum_j)),0) for positive Mij; no smoothing or pruning",
            "svd": {"class": "sklearn.decomposition.TruncatedSVD", "n_components": 32,
                    "random_state": 42, "n_iter": 5, "normalization": "none"},
            "document_embedding": "mean known token embeddings by occurrence; OOV excluded; all-OOV zero",
            "structured": "reuse frozen C2_B structured prefix columns only; exclude frozen TF-IDF and final missing flag",
            "hgb": HGB_PARAMS, "threads": MAX_THREADS,
            "policy": {"champion_present": "0.5*originalUniformPOS + 0.5*originalA02",
                       "candidate_present": "0.75*champion + 0.25*auxiliary",
                       "missing_route": "exact original A02 champion probability; threshold 0.53230976129477",
                       "present_threshold": CHAMPION_THRESHOLD},
        },
        "budget": {"representation_fit_cap": 4, "predictor_fit_cap": 4,
                   "fit_order": [f"E{o}_{arm}" for o, arm in FIT_ORDER],
                   "no_retry_after_started_failure": True},
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": __import__("scipy").__version__,
                             "scikit_learn": __import__("sklearn").__version__},
    }
    body["freeze_signature"] = canonical_sha256(body)
    atomic_json(FREEZE_PATH, body, write_once=True)
    print(json.dumps({"status": "SOURCE_FREEZE_READY", "path": str(FREEZE_PATH),
                      "sha256": sha256_file(FREEZE_PATH), "freeze_signature": body["freeze_signature"],
                      "origins": {key: {"fit_rows": value["fit_rows"],
                                        "score_rows": value["score_rows"],
                                        "vocabulary": value["ordered_unigram_count"],
                                        "structured_width": value["structured_width"]}
                                  for key, value in origins.items()}}, sort_keys=True))


def _verify_freeze(source_root: Path | None = None, audit_root: Path | None = None,
                   *, require_committed: bool = True) -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise PilotError("Committed source freeze is missing")
    freeze = read_json(FREEZE_PATH)
    signature = freeze.get("freeze_signature")
    body = dict(freeze)
    body.pop("freeze_signature", None)
    if signature != canonical_sha256(body):
        raise PilotError("Source freeze signature is invalid")
    if require_committed:
        assert_committed_exact(ROOT, FREEZE_PATH,
                               FREEZE_PATH.relative_to(ROOT).as_posix())
        for relative in freeze["owned_source_paths"]:
            assert_committed_exact(ROOT, ROOT / relative, relative)
    source_root = Path(source_root or freeze["source_root"]).resolve()
    audit_root = Path(audit_root or freeze["audit_root"]).resolve()
    helper = _load_reference_helper(audit_root)
    current_sources = _source_hashes(source_root, audit_root, helper)
    if current_sources != freeze["source_files"]:
        raise PilotError("Frozen external source/input file hashes changed")
    current_owned = _owned_source_hashes()
    if current_owned != freeze["owned_source_sha256"]:
        raise PilotError("Frozen owned source hashes changed after source freeze")
    return freeze


def _append_ledger(event: dict[str, Any]) -> None:
    FIT_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")) + "\n"
    with FIT_LEDGER_PATH.open("a", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_events() -> list[dict[str, Any]]:
    if not FIT_LEDGER_PATH.exists():
        return []
    return [json.loads(line) for line in FIT_LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@contextmanager
def _heavy_job_lock(wait_seconds: int = 60) -> Iterator[dict[str, Any]]:
    lock_path = Path(os.environ.get("TEMP", tempfile.gettempdir())) / "kasdad_single_heavy_job.lock"
    deadline = time.monotonic() + max(0, wait_seconds)
    if lock_path.exists():
        lock_file = lock_path.open("r+b")
    else:
        try:
            lock_file = lock_path.open("x+b")
            lock_file.write(b"\0")
            lock_file.flush()
            os.fsync(lock_file.fileno())
            lock_file.seek(0)
        except FileExistsError:
            lock_file = lock_path.open("r+b")
    locked = False
    while not locked:
        locked = try_advisory_byte_lock(lock_file)
        if locked:
            break
        if time.monotonic() >= deadline:
            lock_file.close()
            raise LockBusy(f"Shared heavy-job byte-zero advisory lock is busy: {lock_path}")
        print(json.dumps({"status": "WAITING_FOR_SHARED_HEAVY_JOB_LOCK",
                          "lock": str(lock_path), "lock_protocol": "byte-zero advisory"}), flush=True)
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    try:
        yield {"pilot_id": "CLS_PPMI_CONTEXT_V1", "pid": os.getpid(),
               "started_at_utc": utc_now(), "path": str(lock_path),
               "protocol": "byte-zero advisory lock"}
    finally:
        if locked:
            release_advisory_byte_lock(lock_file)
        lock_file.close()


class _PeakRss:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRss":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass


def _resource_snapshot(stage: str, *, minimum_available: int) -> dict[str, Any]:
    virtual = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    if int(virtual.available) < minimum_available:
        raise PilotError(f"Available memory below {minimum_available} bytes before {stage}: {virtual.available}")
    return {"stage": stage, "available_bytes": int(virtual.available),
            "rss_bytes": int(rss), "minimum_available_bytes": int(minimum_available)}


def _max_context_pairs(documents: list[list[int]]) -> int:
    count = 0
    for document in documents:
        n = len(document)
        count += 2 * sum(max(n - distance, 0) for distance in (1, 2))
    return int(count)


def _context_memory_estimate(documents: list[list[int]], vocab_size: int) -> dict[str, int]:
    max_entries = _max_context_pairs(documents)
    count_csr = resource_bytes_for_csr(max_entries, vocab_size, data_bytes=4, index_bytes=4)
    ppmi_csr = resource_bytes_for_csr(max_entries, vocab_size, data_bytes=8, index_bytes=4)
    coo_triplets = int(max_entries * 12)
    overhead = 64 * 1024**2
    return {"max_directed_pair_entries": max_entries,
            "count_csr_bytes_upper_bound": count_csr,
            "ppmi_csr_bytes_upper_bound": ppmi_csr,
            "coo_triplet_bytes_upper_bound": coo_triplets,
            "workspace_overhead_bytes": overhead,
            "estimated_peak_bytes": count_csr + ppmi_csr + coo_triplets + overhead}


def _fit_memory_estimate(documents: list[list[int]], vocab_size: int,
                         fit_present: int, score_present: int,
                         structured_width: int) -> dict[str, int]:
    estimate = _context_memory_estimate(documents, vocab_size)
    feature_width = EMBEDDING_DIM + int(structured_width)
    dense_feature_bytes = int((fit_present + score_present) * feature_width * 8)
    embedding_bytes = int(vocab_size * EMBEDDING_DIM * 8)
    hgb_workspace_bytes = 256 * 1024**2
    total = estimate["estimated_peak_bytes"] + dense_feature_bytes * 3 + embedding_bytes + hgb_workspace_bytes
    return {**estimate, "dense_hgb_feature_bytes": dense_feature_bytes,
            "term_embedding_bytes": embedding_bytes,
            "hgb_workspace_bytes": hgb_workspace_bytes,
            "estimated_peak_bytes": int(total)}


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PilotError(f"Write-once NPZ already exists: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PilotError(f"Write-once sparse artifact already exists: {path}")
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    sparse.save_npz(temporary, sparse.csr_matrix(matrix), compressed=True)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PilotError(f"Write-once estimator artifact already exists: {path}")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    joblib.dump(value, temporary, compress=3)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _positive_probability(model: HistGradientBoostingClassifier, features: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == 1)
    if len(positions) != 1:
        raise PilotError(f"HGB positive class 1 missing from classes={classes.tolist()}")
    values = np.asarray(model.predict_proba(features)[:, int(positions[0])], dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise PilotError("HGB produced invalid probabilities")
    return values


def _load_contexts_for_fit(freeze: dict[str, Any], source_root: Path,
                           audit_root: Path) -> tuple[Any, pd.DataFrame, dict[int, dict[str, Any]]]:
    helper = _load_reference_helper(audit_root)
    train_path = source_root / TRAIN_REL
    if sha256_file(train_path) != TRAIN_SHA256:
        raise PilotError("Official Train changed before representation/predictor fit")
    frame = pd.read_csv(train_path, usecols=["record_id", "cycle_code", "signal_stream"],
                        dtype={"record_id": "string"})
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    contexts: dict[int, dict[str, Any]] = {}
    for origin in ORIGINS:
        context = helper._load_origin_context(source_root, origin, require_fit_labels=True)
        fit_cycles = frame.iloc[context["fit_rows"]]["cycle_code"].to_numpy(dtype=np.int64)
        assert_prefix_cycles(fit_cycles, origin)
        y_fit = np.asarray(context["y_fit"], dtype=np.int64).reshape(-1)
        if len(y_fit) != len(context["fit_rows"]) or not np.isin(y_fit, (0, 1)).all():
            raise PilotError(f"Fit labels for E{origin} are not aligned binary 0/1 values")
        context["y_fit"] = y_fit
        contexts[origin] = context
    return helper, frame, contexts


def command_preflight() -> None:
    freeze = _verify_freeze(require_committed=True)
    if PREFLIGHT_PATH.exists():
        raise PilotError(f"Write-once fit preflight already exists: {PREFLIGHT_PATH}")
    source_root = Path(freeze["source_root"])
    audit_root = Path(freeze["audit_root"])
    helper = _load_reference_helper(audit_root)
    train_path = source_root / TRAIN_REL
    if sha256_file(train_path) != TRAIN_SHA256:
        raise PilotError("Official source changed before fit resource preflight")
    frame = pd.read_csv(train_path, usecols=["record_id", "cycle_code", "signal_stream"],
                        dtype={"record_id": "string"})
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    origins: dict[str, Any] = {}
    all_pass = True
    for origin in ORIGINS:
        context = helper._load_origin_context(source_root, origin, require_fit_labels=False)
        model = joblib.load(Path(freeze["origins"][str(origin)]["model_path"]))
        vectorizer = model.text_space.vectorizer
        terms = ordered_unigrams(vectorizer)
        fit_rows = np.asarray(context["fit_rows"], dtype=np.int64)
        score_rows = np.asarray(context["score_rows"], dtype=np.int64)
        fit_frame = frame.iloc[fit_rows]
        score_frame = frame.iloc[score_rows]
        _, fit_missing = model.text_space.transform(fit_frame["signal_stream"])
        _, score_missing = model.text_space.transform(score_frame["signal_stream"])
        fit_present = ~np.asarray(fit_missing, dtype=bool)
        score_present = ~np.asarray(score_missing, dtype=bool)
        fit_documents, _, fit_stats = tokenize_streams(
            fit_frame["signal_stream"].tolist(), vectorizer,
            {term: index for index, term in enumerate(terms)}, fit_missing)
        score_documents, _, score_stats = tokenize_streams(
            score_frame["signal_stream"].tolist(), vectorizer,
            {term: index for index, term in enumerate(terms)}, score_missing)
        graph_documents = [doc for doc, present in zip(fit_documents, fit_present, strict=True) if present]
        memory = _fit_memory_estimate(graph_documents, len(terms), int(fit_present.sum()),
                                       int(score_present.sum()), int(freeze["origins"][str(origin)]["structured_width"]))
        available = int(psutil.virtual_memory().available)
        feasible = available >= MIN_START_AVAILABLE and memory["estimated_peak_bytes"] <= available - MIN_RESERVE_AVAILABLE
        all_pass = all_pass and feasible
        origins[str(origin)] = {
            "fit_rows": int(len(fit_rows)), "fit_present": int(fit_present.sum()),
            "fit_missing": int((~fit_present).sum()), "score_rows": int(len(score_rows)),
            "score_present": int(score_present.sum()), "score_missing": int((~score_present).sum()),
            "vocabulary_size": int(len(terms)), "fit_token_stats": fit_stats,
            "score_token_stats": score_stats, "memory_estimate": memory,
            "available_bytes_at_preflight": available,
            "minimum_start_available_bytes": MIN_START_AVAILABLE,
            "minimum_reserve_bytes": MIN_RESERVE_AVAILABLE,
            "feasible_under_current_memory": bool(feasible),
            "fit_row_ids_sha256": helper.sha256_array(context["fit_ids"]),
            "score_row_ids_sha256": helper.sha256_array(context["score_ids"]),
            "manifest_sha256": context["manifest_sha256"],
        }
    source_files_unchanged = _source_hashes(source_root, audit_root, helper) == freeze["source_files"]
    all_pass = all_pass and source_files_unchanged
    status = "PASS" if all_pass else (
        "BLOCKED_RESOURCE_PREFLIGHT" if source_files_unchanged else "BLOCKED_SOURCE_CHANGED")
    body = {
        "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
        "freeze_signature": freeze["freeze_signature"], "status": status,
        "origins": origins, "representation_fit_cap": 4, "predictor_fit_cap": 4,
        "fit_order": [f"E{o}_{arm}" for o, arm in FIT_ORDER],
        "query_targets_read": False, "test_or_sample_opened": False,
        "source_files_unchanged": source_files_unchanged,
        "created_at_utc": utc_now(),
    }
    body["preflight_signature"] = canonical_sha256(body)
    atomic_json(PREFLIGHT_PATH, body, write_once=True)
    print(json.dumps({"status": status, "path": str(PREFLIGHT_PATH),
                      "sha256": sha256_file(PREFLIGHT_PATH),
                      "origins": {origin: {"feasible": item["feasible_under_current_memory"],
                                           "estimated_peak_bytes": item["memory_estimate"]["estimated_peak_bytes"],
                                           "available_bytes": item["available_bytes_at_preflight"]}
                                  for origin, item in origins.items()}}, sort_keys=True))


def _verify_preflight(freeze: dict[str, Any], *, require_committed: bool = True) -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise PilotError("Committed resource preflight is required before official fits")
    preflight = read_json(PREFLIGHT_PATH)
    body = dict(preflight)
    signature = body.pop("preflight_signature", None)
    if signature != canonical_sha256(body):
        raise PilotError("Fit preflight signature is invalid")
    if preflight.get("freeze_signature") != freeze["freeze_signature"] \
            or preflight.get("status") != "PASS" \
            or preflight.get("source_files_unchanged") is not True \
            or preflight.get("query_targets_read") is not False:
        raise PilotError("Fit preflight is blocked or bound to a different freeze")
    if set(preflight.get("origins", {})) != {"67", "75"}:
        raise PilotError("Fit preflight does not cover E67 and E75")
    expected_fit_order = [f"E{o}_{arm}" for o, arm in FIT_ORDER]
    if preflight.get("representation_fit_cap") != 4 \
            or preflight.get("predictor_fit_cap") != 4 \
            or preflight.get("fit_order") != expected_fit_order \
            or any(item.get("feasible_under_current_memory") is not True
                   for item in preflight["origins"].values()):
        raise PilotError("Fit preflight does not authorize exactly the frozen four-fit plan")
    if require_committed:
        assert_committed_exact(ROOT, PREFLIGHT_PATH, PREFLIGHT_PATH.relative_to(ROOT).as_posix())
    return preflight


def _fit_one(origin: int, arm: str, context: dict[str, Any], frame: pd.DataFrame,
             freeze: dict[str, Any], vectorizer: Any, fit_documents: list[list[int]],
             shuffled_documents: list[list[int]], score_documents: list[list[int]],
             fit_present: np.ndarray, score_present: np.ndarray,
             structured_fit: sparse.csr_matrix, structured_score: sparse.csr_matrix,
             y_fit: np.ndarray, unigram_count: int,
             lock_info: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    fit_id = f"CLS_PPMI_CONTEXT_V1_E{origin}_{arm}"
    graph_docs_all = fit_documents if arm == "REAL" else shuffled_documents
    graph_documents = [doc for doc, is_present in zip(graph_docs_all, fit_present, strict=True)
                       if bool(is_present)]
    memory = _context_memory_estimate(graph_documents, unigram_count)
    available_before = int(psutil.virtual_memory().available)
    if available_before < MIN_START_AVAILABLE:
        raise PilotError(f"Available memory below 1 GiB before {fit_id} representation fit")
    if memory["estimated_peak_bytes"] > available_before - MIN_RESERVE_AVAILABLE:
        raise PilotError(f"Sparse context upper-bound estimate exceeds available memory reserve before {fit_id}: {memory}")
    source_root = Path(freeze["source_root"])
    audit_root = Path(freeze["audit_root"])
    current_external_hashes = _source_hashes(source_root, audit_root)
    if current_external_hashes != freeze["source_files"]:
        raise PilotError(f"Frozen external source/input hashes changed before {fit_id}")
    _append_ledger({"event": "REPRESENTATION_FIT_STARTED", "fit_id": fit_id,
                    "origin": origin, "arm": arm, "freeze_signature": freeze["freeze_signature"],
                    "source_commit": git("rev-parse", "HEAD"), "time_utc": utc_now(),
                    "lock": lock_info, "resource_before": _resource_snapshot(
                        f"{fit_id} representation", minimum_available=MIN_START_AVAILABLE),
                    "memory_estimate": memory})
    start = time.perf_counter()
    try:
        counts = cooccurrence_counts(graph_documents, unigram_count)
        ppmi = positive_ppmi(counts)
        active_rows = int(np.count_nonzero(np.asarray(ppmi.getnnz(axis=1)).reshape(-1)))
        active_cols = int(np.count_nonzero(np.asarray(ppmi.getnnz(axis=0)).reshape(-1)))
        if ppmi.nnz == 0 or min(active_rows, active_cols) < 33:
            raise PilotError(f"PPMI graph has insufficient active dimensions before HGB fit: rows={active_rows}, cols={active_cols}, nnz={ppmi.nnz}")
        svd = TruncatedSVD(n_components=EMBEDDING_DIM, random_state=42, n_iter=5)
        with threadpool_limits(limits=MAX_THREADS), _PeakRss() as representation_rss:
            term_embeddings = np.asarray(svd.fit_transform(ppmi), dtype=np.float64)
        representation_seconds = float(time.perf_counter() - start)
        if term_embeddings.shape != (unigram_count, EMBEDDING_DIM) or not np.isfinite(term_embeddings).all():
            raise PilotError(f"SVD output has unexpected shape or nonfinite values for {fit_id}")
        runtime_dir = RUNTIME_ROOT / f"E{origin}" / arm
        _atomic_sparse(runtime_dir / "counts.npz", counts)
        _atomic_sparse(runtime_dir / "ppmi.npz", ppmi)
        _atomic_npz(runtime_dir / "svd_state.npz", term_embeddings=term_embeddings,
                    components=np.asarray(svd.components_, dtype=np.float64),
                    singular_values=np.asarray(svd.singular_values_, dtype=np.float64),
                    explained_variance_ratio=np.asarray(svd.explained_variance_ratio_, dtype=np.float64))
        _append_ledger({"event": "REPRESENTATION_FIT_SAVED", "fit_id": fit_id,
                        "origin": origin, "arm": arm, "time_utc": utc_now(),
                        "runtime_seconds": representation_seconds,
                        "counts_nnz": int(counts.nnz), "ppmi_nnz": int(ppmi.nnz),
                        "active_rows": active_rows, "active_cols": active_cols,
                        "svd_embedding_sha256": sha256_array(term_embeddings),
                        "resource_after": _resource_snapshot(
                            f"{fit_id} representation", minimum_available=MIN_RESERVE_AVAILABLE),
                        "peak_rss_bytes": int(representation_rss.peak)})
        means_fit, fit_coverage = document_means(fit_documents, term_embeddings, ~fit_present)
        means_score, score_coverage = document_means(score_documents, term_embeddings, ~score_present)
        X_fit = np.concatenate((means_fit[fit_present],
                                structured_fit[fit_present].toarray()), axis=1)
        X_score = np.concatenate((means_score[score_present],
                                  structured_score[score_present].toarray()), axis=1)
        if X_fit.shape[1] != EMBEDDING_DIM + structured_fit.shape[1] \
                or X_score.shape[1] != X_fit.shape[1]:
            raise PilotError(f"Supervised feature width mismatch for {fit_id}")
        y_present = y_fit[fit_present]
        class_counts = {str(int(label)): int(np.count_nonzero(y_present == label))
                        for label in np.unique(y_present)}
        if set(np.unique(y_present)) != {0, 1}:
            raise PilotError(f"Present-stream training rows lack a binary class for {fit_id}")
        resource_before_hgb = _resource_snapshot(f"{fit_id} HGB", minimum_available=MIN_START_AVAILABLE)
        _append_ledger({"event": "PREDICTOR_FIT_STARTED", "fit_id": fit_id,
                        "origin": origin, "arm": arm, "time_utc": utc_now(),
                        "fit_rows_present": int(fit_present.sum()),
                        "fit_class_counts_present": class_counts,
                        "feature_width": int(X_fit.shape[1]),
                        "resource_before": resource_before_hgb})
        hgb_start = time.perf_counter()
        classifier = HistGradientBoostingClassifier(**HGB_PARAMS)
        with threadpool_limits(limits=MAX_THREADS), _PeakRss() as hgb_rss:
            classifier.fit(X_fit, y_present)
            p_fit = _positive_probability(classifier, X_fit)
            p_score = _positive_probability(classifier, X_score)
        hgb_seconds = float(time.perf_counter() - hgb_start)
        fit_decision = (p_fit >= CHAMPION_THRESHOLD).astype(np.int64)
        fit_quality = binary_metrics(y_present, p_fit, fit_decision)
        model_path = runtime_dir / "hgb.joblib"
        _atomic_joblib(model_path, classifier)
        npz_paths = [runtime_dir / "counts.npz", runtime_dir / "ppmi.npz", runtime_dir / "svd_state.npz"]
        artifact_hashes = {str(path.relative_to(ROOT).as_posix()): sha256_file(path)
                           for path in [*npz_paths, model_path]}
        fit_summary = {
            "fit_id": fit_id, "origin": origin, "arm": arm,
            "fit_rows_total": int(len(context["fit_rows"])),
            "fit_rows_present": int(fit_present.sum()), "fit_rows_missing": int((~fit_present).sum()),
            "fit_class_counts_present": class_counts,
            "fit_cycle_min": int(frame.iloc[context["fit_rows"]]["cycle_code"].min()),
            "fit_cycle_max": int(frame.iloc[context["fit_rows"]]["cycle_code"].max()),
            "score_rows": int(len(context["score_rows"])),
            "score_present": int(score_present.sum()), "score_missing": int((~score_present).sum()),
            "vocabulary_size": int(unigram_count), "context_window_distances": [1, 2],
            "directed_pair_weight": 1, "pair_count_total_directed": int(counts.sum()),
            "counts_shape": list(counts.shape), "counts_nnz": int(counts.nnz),
            "counts_density": float(counts.nnz / (counts.shape[0] * counts.shape[1])),
            "ppmi_nnz": int(ppmi.nnz),
            "ppmi_density": float(ppmi.nnz / (ppmi.shape[0] * ppmi.shape[1])),
            "active_rows": active_rows, "active_cols": active_cols,
            "active_dimensions_gate_pass": bool(ppmi.nnz > 0 and min(active_rows, active_cols) >= 33),
            "svd_components": int(EMBEDDING_DIM), "svd_singular_values": [float(x) for x in svd.singular_values_],
            "svd_embedding_sha256": sha256_array(term_embeddings),
            "document_fit_coverage": fit_coverage, "document_score_coverage": score_coverage,
            "svd_runtime_seconds": representation_seconds, "hgb_runtime_seconds": hgb_seconds,
            "peak_rss_representation_bytes": int(representation_rss.peak),
            "peak_rss_predictor_bytes": int(hgb_rss.peak),
            "hgb_params": HGB_PARAMS, "fit_quality": fit_quality,
            "artifact_sha256": artifact_hashes,
            "query_targets_read": False, "test_or_sample_opened": False,
            "runtime_path": str(runtime_dir.relative_to(ROOT).as_posix()),
        }
        _append_ledger({"event": "PREDICTOR_FIT_SAVED", "fit_id": fit_id,
                        "origin": origin, "arm": arm, "time_utc": utc_now(),
                        "runtime_seconds": hgb_seconds, "artifact_sha256": artifact_hashes,
                        "fit_quality": fit_quality,
                        "resource_after": _resource_snapshot(
                            f"{fit_id} predictor", minimum_available=MIN_RESERVE_AVAILABLE),
                        "peak_rss_bytes": int(hgb_rss.peak)})
        return fit_summary, p_score
    except Exception as exc:
        _append_ledger({"event": "FIT_FAILED", "fit_id": fit_id, "origin": origin,
                        "arm": arm, "time_utc": utc_now(),
                        "error_type": type(exc).__name__, "error": str(exc)[:1000],
                        "representation_started": True,
                        "predictor_started": any(event.get("fit_id") == fit_id and
                                                 event.get("event") == "PREDICTOR_FIT_STARTED"
                                                 for event in _ledger_events())})
        raise


def _write_predictions(origin: int, context: dict[str, Any], frame: pd.DataFrame,
                       score_present: np.ndarray, p_real: np.ndarray,
                       p_shuffled: np.ndarray, fit_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    p_a02 = np.asarray(context["p_a02"], dtype=np.float64).reshape(-1)
    p_pos = np.asarray(context["p_pos_original"], dtype=np.float64).reshape(-1)
    present = np.asarray(score_present, dtype=bool).reshape(-1)
    p_champion = np.where(present, 0.5 * (p_pos + p_a02), p_a02)
    p_candidate = p_champion.copy()
    p_control = p_champion.copy()
    p_candidate[present] = 0.75 * p_champion[present] + 0.25 * p_real[present]
    p_control[present] = 0.75 * p_champion[present] + 0.25 * p_shuffled[present]
    if not np.array_equal(p_candidate[~present], p_champion[~present]) \
            or not np.array_equal(p_control[~present], p_champion[~present]):
        raise PilotError(f"Missing-stream route differs from champion for E{origin}")
    thresholds = np.where(present, CHAMPION_THRESHOLD, MISSING_THRESHOLD).astype(np.float64)
    frame_rows = np.asarray(context["score_rows"], dtype=np.int64)
    frame_ids = frame.iloc[frame_rows]["record_id"].astype(str).to_numpy(dtype=str)
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    arrays = {
        "origin": np.asarray([origin], dtype=np.int64),
        "score_rows": frame_rows,
        "score_record_ids": frame_ids,
        "score_cycles": cycles,
        "present_mask": present,
        "p_champion": p_champion,
        "p_aux_real": np.where(present, p_real, 0.0),
        "p_aux_shuffled": np.where(present, p_shuffled, 0.0),
        "p_candidate_real": p_candidate,
        "p_control_shuffled": p_control,
        "decision_thresholds": thresholds,
        "entry_friction": np.asarray(context["entry_friction"]).astype(str),
        "text_load_group": np.asarray(context["text_load_group"]).astype(str),
        "fit_row_ids_sha256": np.asarray([sha256_array(context["fit_ids"])], dtype=str),
        "matrix_manifest_sha256": np.asarray([context["manifest_sha256"]], dtype=str),
    }
    path = EXPERIMENT_ROOT / f"predictions_E{origin}.npz"
    _atomic_npz(path, **arrays)
    return {"origin": origin, "path": str(path.relative_to(ROOT).as_posix()),
            "sha256": sha256_file(path), "score_rows": int(len(frame_rows)),
            "score_present": int(present.sum()), "score_missing": int((~present).sum()),
            "fit_ids_sha256": sha256_array(context["fit_ids"]),
            "matrix_manifest_sha256": context["manifest_sha256"],
            "score_ids_sha256": sha256_array(frame_ids),
            "fit_ids": [summary["fit_id"] for summary in fit_summaries]}


def _verify_prelabel_artifacts(freeze: dict[str, Any], *, require_committed: bool) -> dict[str, Any]:
    if not PRELABEL_PATH.is_file() or not PRELABEL_VERIFICATION_PATH.is_file():
        raise PilotError("Prelabel predictions or verification receipt is missing")
    prelabel = read_json(PRELABEL_PATH)
    body = dict(prelabel)
    signature = body.pop("prelabel_signature", None)
    if signature != canonical_sha256(body):
        raise PilotError("Prelabel signature is invalid")
    if prelabel.get("freeze_signature") != freeze["freeze_signature"]:
        raise PilotError("Prelabel was built from a different source freeze")
    if prelabel.get("query_targets_read") is not False or prelabel.get("test_or_sample_opened") is not False:
        raise PilotError("Prelabel boundary reports query/test access")
    if len(prelabel.get("fit_ids", [])) != 4 or len(prelabel.get("predictions", [])) != 2:
        raise PilotError("Prelabel manifest does not contain exactly four fits and two origins")
    if not FIT_SUMMARY_PATH.is_file() or not FIT_LEDGER_PATH.is_file():
        raise PilotError("Fit summary or fit ledger is missing before prelabel verification")
    if sha256_file(FIT_SUMMARY_PATH) != prelabel.get("fit_summary_sha256") \
            or sha256_file(FIT_LEDGER_PATH) != prelabel.get("fit_ledger_sha256"):
        raise PilotError("Fit summary/ledger hashes differ from prelabel receipt")
    fit_summary = read_json(FIT_SUMMARY_PATH)
    expected_fit_ids = [f"CLS_PPMI_CONTEXT_V1_E{o}_{arm}" for o, arm in FIT_ORDER]
    if fit_summary.get("fit_ids") != expected_fit_ids \
            or prelabel.get("fit_ids") != expected_fit_ids \
            or fit_summary.get("representation_fit_count") != 4 \
            or fit_summary.get("predictor_fit_count") != 4:
        raise PilotError("Fit summary does not show exactly the four frozen starts")
    events = _ledger_events()
    expected_ids = set(expected_fit_ids)
    for event_name in ("REPRESENTATION_FIT_STARTED", "REPRESENTATION_FIT_SAVED",
                       "PREDICTOR_FIT_STARTED", "PREDICTOR_FIT_SAVED"):
        seen = [event.get("fit_id") for event in events if event.get("event") == event_name]
        if len(seen) != 4 or set(seen) != expected_ids:
            raise PilotError(f"Fit ledger has an invalid {event_name} count")
    if any(event.get("event") == "FIT_FAILED" for event in events):
        raise PilotError("Fit ledger contains a failed/interrupted official fit")
    if require_committed:
        assert_committed_exact(ROOT, FIT_SUMMARY_PATH, FIT_SUMMARY_PATH.relative_to(ROOT).as_posix())
        assert_committed_exact(ROOT, FIT_LEDGER_PATH, FIT_LEDGER_PATH.relative_to(ROOT).as_posix())
    source_root = Path(freeze["source_root"])
    train_path = source_root / TRAIN_REL
    if sha256_file(train_path) != TRAIN_SHA256:
        raise PilotError("Official source changed before prelabel identity verification")
    identity = pd.read_csv(train_path, usecols=["record_id", "cycle_code"],
                           dtype={"record_id": "string"})
    identity["cycle_code"] = pd.to_numeric(identity["cycle_code"], errors="raise").astype(np.int64)
    entries: dict[int, Any] = {}
    checks: dict[str, Any] = {}
    for item in prelabel["predictions"]:
        origin = int(item["origin"])
        path = ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise PilotError(f"Prelabel prediction hash mismatch for E{origin}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"origin", "score_rows", "score_record_ids", "score_cycles", "present_mask",
                        "p_champion", "p_aux_real", "p_aux_shuffled", "p_candidate_real",
                        "p_control_shuffled", "decision_thresholds", "entry_friction", "text_load_group",
                        "fit_row_ids_sha256", "matrix_manifest_sha256"}
            if not required.issubset(archive.files) or any("label" in key.lower() for key in archive.files):
                raise PilotError(f"Prelabel prediction schema has missing or label keys for E{origin}")
            values = {key: archive[key].copy() for key in required}
        if int(values["origin"].reshape(-1)[0]) != origin:
            raise PilotError(f"Prelabel origin mismatch for E{origin}")
        n = len(values["score_rows"])
        if any(len(values[key]) != n for key in ("score_record_ids", "score_cycles", "present_mask",
                                                  "p_champion", "p_aux_real", "p_aux_shuffled",
                                                  "p_candidate_real", "p_control_shuffled",
                                                  "decision_thresholds", "entry_friction", "text_load_group")):
            raise PilotError(f"Prelabel prediction rows are misaligned for E{origin}")
        present = values["present_mask"].astype(bool)
        if not np.array_equal(values["p_candidate_real"][~present], values["p_champion"][~present]) \
                or not np.array_equal(values["p_control_shuffled"][~present], values["p_champion"][~present]):
            raise PilotError(f"Prelabel missing-stream route is not exact for E{origin}")
        for key in ("p_champion", "p_candidate_real", "p_control_shuffled"):
            prob = np.asarray(values[key], dtype=np.float64)
            if not np.isfinite(prob).all() or np.any((prob < 0.0) | (prob > 1.0)):
                raise PilotError(f"Invalid probabilities in prelabel key {key} for E{origin}")
        expected_threshold = np.where(present, CHAMPION_THRESHOLD, MISSING_THRESHOLD)
        if not np.array_equal(values["decision_thresholds"], expected_threshold):
            raise PilotError(f"Fixed decision thresholds changed for E{origin}")
        rows = values["score_rows"].astype(np.int64)
        source_ids = identity.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
        source_cycles = identity.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
        if not np.array_equal(values["score_record_ids"].astype(str), source_ids) \
                or not np.array_equal(values["score_cycles"].astype(np.int64), source_cycles):
            raise PilotError(f"Prelabel score IDs/cycles differ from official Train identity for E{origin}")
        if sha256_array(values["score_record_ids"].astype(str)) != item["score_ids_sha256"]:
            raise PilotError(f"Prelabel score ID digest differs from receipt for E{origin}")
        if item["matrix_manifest_sha256"] != freeze["origins"][str(origin)]["manifest_sha256"]:
            raise PilotError(f"Prelabel matrix manifest binding differs for E{origin}")
        if values["fit_row_ids_sha256"].reshape(-1)[0] != freeze["origins"][str(origin)]["fit_row_ids_sha256"]:
            raise PilotError(f"Prelabel fit ID binding differs from source freeze for E{origin}")
        entries[origin] = values
        checks[str(origin)] = {"rows": n, "present": int(present.sum()),
                               "missing": int((~present).sum()),
                               "missing_route_exact": True,
                               "finite_probability_arrays": True}
    if set(entries) != set(ORIGINS):
        raise PilotError("Prelabel must contain exactly E67 and E75 predictions")
    primary_ids = np.concatenate((entries[67]["score_record_ids"][
        (entries[67]["score_cycles"] >= 68) & (entries[67]["score_cycles"] <= 75)],
        entries[75]["score_record_ids"][
        (entries[75]["score_cycles"] >= 76) & (entries[75]["score_cycles"] <= 79)]))
    if len(primary_ids) != 20_903 or len(set(primary_ids.astype(str))) != 20_903:
        raise PilotError("Unique primary prelabel rows are not exactly 20,903 unique records")
    checks["primary_unique_rows"] = {"rows": int(len(primary_ids)), "unique_ids": int(len(set(primary_ids.astype(str))))}
    if require_committed:
        assert_committed_exact(ROOT, PRELABEL_PATH, PRELABEL_PATH.relative_to(ROOT).as_posix())
        assert_committed_exact(ROOT, PRELABEL_VERIFICATION_PATH,
                               PRELABEL_VERIFICATION_PATH.relative_to(ROOT).as_posix())
        for item in prelabel["predictions"]:
            path = ROOT / item["path"]
            assert_committed_exact(ROOT, path, path.relative_to(ROOT).as_posix())
    return {"prelabel": prelabel, "predictions": entries, "checks": checks}


def command_fit(lock_wait_seconds: int = 60) -> None:
    freeze = _verify_freeze(require_committed=True)
    _verify_preflight(freeze, require_committed=True)
    if FIT_LEDGER_PATH.exists() or FIT_SUMMARY_PATH.exists() or PRELABEL_PATH.exists():
        raise PilotError("Fit ledger/output already exists; no automatic retry or fit-cap expansion")
    if not (EXPERIMENT_ROOT / ".gitignore").is_file():
        raise PilotError("Runtime experiment .gitignore must be present before artifact creation")
    assert_committed_exact(ROOT, EXPERIMENT_ROOT / ".gitignore",
                           (EXPERIMENT_ROOT / ".gitignore").relative_to(ROOT).as_posix())
    source_root = Path(freeze["source_root"])
    audit_root = Path(freeze["audit_root"])
    helper = _load_reference_helper(audit_root)
    with _heavy_job_lock(lock_wait_seconds) as lock_info:
        current_hashes = _source_hashes(source_root, audit_root, helper)
        if current_hashes != freeze["source_files"]:
            raise PilotError("Frozen external source/input hashes changed at fit start")
        helper, feature_frame, contexts = _load_contexts_for_fit(freeze, source_root, audit_root)
        resource_initial = _resource_snapshot("campaign initial after read-only source load",
                                              minimum_available=MIN_START_AVAILABLE)
        _append_ledger({"event": "FIT_CAP_RESERVED", "pilot_id": "CLS_PPMI_CONTEXT_V1",
                        "freeze_signature": freeze["freeze_signature"],
                        "fit_ids": [f"CLS_PPMI_CONTEXT_V1_E{o}_{arm}" for o, arm in FIT_ORDER],
                        "representation_fit_cap": 4, "predictor_fit_cap": 4,
                        "resource_initial": resource_initial, "lock": lock_info,
                        "time_utc": utc_now()})
        fit_label_counts = {}
        for origin, context in contexts.items():
            y = context["y_fit"]
            fit_label_counts[str(origin)] = {"rows": int(len(y)),
                                             "negative": int(np.count_nonzero(y == 0)),
                                             "positive": int(np.count_nonzero(y == 1))}
        fit_summaries: list[dict[str, Any]] = []
        prediction_arrays: dict[int, dict[str, Any]] = {}
        for origin in ORIGINS:
            context = contexts[origin]
            origin_freeze = freeze["origins"][str(origin)]
            model_path = Path(origin_freeze["model_path"])
            model = joblib.load(model_path)
            vectorizer = model.text_space.vectorizer
            terms = ordered_unigrams(vectorizer)
            unigram_to_index = {term: index for index, term in enumerate(terms)}
            fit_frame = feature_frame.iloc[np.asarray(context["fit_rows"], dtype=np.int64)]
            score_frame = feature_frame.iloc[np.asarray(context["score_rows"], dtype=np.int64)]
            _, fit_missing = model.text_space.transform(fit_frame["signal_stream"])
            _, score_missing = model.text_space.transform(score_frame["signal_stream"])
            fit_present = ~np.asarray(fit_missing, dtype=bool)
            score_present = ~np.asarray(score_missing, dtype=bool)
            if not np.array_equal(score_present, np.asarray(context["present"], dtype=bool)):
                raise PilotError(f"Frozen stream availability changed for E{origin}")
            fit_cycles = feature_frame.iloc[context["fit_rows"]]["cycle_code"].to_numpy(dtype=np.int64)
            assert_prefix_cycles(fit_cycles, origin)
            fit_documents, _, fit_token_stats = tokenize_streams(
                fit_frame["signal_stream"].tolist(), vectorizer, unigram_to_index, fit_missing)
            shuffled_documents = shuffle_documents(fit_documents, seed=SHUFFLE_SEED)
            score_documents, _, score_token_stats = tokenize_streams(
                score_frame["signal_stream"].tolist(), vectorizer, unigram_to_index, score_missing)
            width = int(origin_freeze["structured_width"])
            fit_base = sparse.csr_matrix(context["matrices"]["fit_base"], dtype=np.float64)
            score_base = sparse.csr_matrix(context["matrices"]["score_base"], dtype=np.float64)
            structured_fit = fit_base[:, :width]
            structured_score = score_base[:, :width]
            if structured_fit.shape[1] != width or structured_score.shape[1] != width:
                raise PilotError(f"Frozen C2_B structured width mismatch at E{origin}")
            p_real = np.zeros(len(context["score_rows"]), dtype=np.float64)
            p_shuffled = np.zeros(len(context["score_rows"]), dtype=np.float64)
            origin_summaries: list[dict[str, Any]] = []
            for fit_origin, arm in FIT_ORDER:
                if fit_origin != origin:
                    continue
                summary, probabilities = _fit_one(
                    origin, arm, context, feature_frame, freeze, vectorizer,
                    fit_documents, shuffled_documents, score_documents,
                    fit_present, score_present, structured_fit, structured_score,
                    context["y_fit"], len(terms), lock_info)
                summary["fit_token_stats"] = fit_token_stats
                summary["score_token_stats"] = score_token_stats
                fit_summaries.append(summary)
                origin_summaries.append(summary)
                if arm == "REAL":
                    p_real[score_present] = probabilities
                else:
                    p_shuffled[score_present] = probabilities
                _resource_snapshot(f"after E{origin}_{arm}", minimum_available=MIN_RESERVE_AVAILABLE)
            prediction_entry = _write_predictions(origin, context, feature_frame, score_present,
                                                 p_real, p_shuffled, origin_summaries)
            prediction_arrays[origin] = prediction_entry
        external_after = _source_hashes(source_root, audit_root, helper)
        if external_after != freeze["source_files"]:
            raise PilotError("Frozen source/input hashes changed during official fits")
        events = _ledger_events()
        expected_ids = {f"CLS_PPMI_CONTEXT_V1_E{o}_{arm}" for o, arm in FIT_ORDER}
        rep_started = {event["fit_id"] for event in events if event["event"] == "REPRESENTATION_FIT_STARTED"}
        pred_started = {event["fit_id"] for event in events if event["event"] == "PREDICTOR_FIT_STARTED"}
        rep_saved = {event["fit_id"] for event in events if event["event"] == "REPRESENTATION_FIT_SAVED"}
        pred_saved = {event["fit_id"] for event in events if event["event"] == "PREDICTOR_FIT_SAVED"}
        failures = [event for event in events if event["event"] == "FIT_FAILED"]
        if failures or rep_started != expected_ids or pred_started != expected_ids \
                or rep_saved != expected_ids or pred_saved != expected_ids:
            raise PilotError("Fit ledger does not show exactly four saved representations and predictors")
        summary = {
            "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
            "freeze_signature": freeze["freeze_signature"],
            "source_worktree_head": git("rev-parse", "HEAD"),
            "fit_ids": [f"CLS_PPMI_CONTEXT_V1_E{o}_{arm}" for o, arm in FIT_ORDER],
            "fit_label_counts": fit_label_counts, "fit_summaries": fit_summaries,
            "predictions": list(prediction_arrays.values()),
            "representation_fit_count": len(rep_saved), "predictor_fit_count": len(pred_saved),
            "query_targets_read": False, "test_or_sample_opened": False,
            "external_source_sha256_unchanged": True,
            "created_at_utc": utc_now(),
        }
        atomic_json(FIT_SUMMARY_PATH, summary, write_once=True)
        prelabel = {
            "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
            "freeze_signature": freeze["freeze_signature"],
            "source_worktree_head": git("rev-parse", "HEAD"),
            "fit_ids": summary["fit_ids"], "representation_fit_count": 4,
            "predictor_fit_count": 4, "query_targets_read": False,
            "test_or_sample_opened": False, "predictions": list(prediction_arrays.values()),
            "fit_summary_sha256": sha256_file(FIT_SUMMARY_PATH),
            "fit_ledger_sha256": sha256_file(FIT_LEDGER_PATH),
            "created_at_utc": utc_now(),
        }
        prelabel["prelabel_signature"] = canonical_sha256(prelabel)
        atomic_json(PRELABEL_PATH, prelabel, write_once=True)
        prelabel_check = _verify_prelabel_artifacts(freeze, require_committed=False)
        verification = {
            "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
            "status": "PASS", "freeze_signature": freeze["freeze_signature"],
            "prelabel_signature": prelabel["prelabel_signature"],
            "checks": prelabel_check["checks"], "query_targets_read": False,
            "test_or_sample_opened": False,
        }
        atomic_json(PRELABEL_VERIFICATION_PATH, verification, write_once=True)
        print(json.dumps({"status": "PRELABEL_READY_PENDING_COMMIT",
                          "fit_summary": str(FIT_SUMMARY_PATH),
                          "prelabel": str(PRELABEL_PATH),
                          "prelabel_verification": str(PRELABEL_VERIFICATION_PATH),
                          "fit_ids": summary["fit_ids"],
                          "prediction_hashes": [{"origin": item["origin"], "sha256": item["sha256"]}
                                                for item in summary["predictions"]],
                          "query_targets_read": False,
                          "next": "Commit the named prelabel outputs before running evaluate."},
                         sort_keys=True))


def _metric_bundle(y: np.ndarray, prediction: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    subset_y = np.asarray(y[mask], dtype=np.int64)
    thresholds = np.asarray(prediction["decision_thresholds"][mask], dtype=np.float64)
    probabilities = {
        "champion": np.asarray(prediction["p_champion"][mask], dtype=np.float64),
        "candidate_real": np.asarray(prediction["p_candidate_real"][mask], dtype=np.float64),
        "control_shuffled": np.asarray(prediction["p_control_shuffled"][mask], dtype=np.float64),
    }
    result = {name: binary_metrics(subset_y, values, (values >= thresholds).astype(np.int64))
              for name, values in probabilities.items()}
    result["gains"] = {
        "candidate_real_minus_champion_f1": float(result["candidate_real"]["f1"] - result["champion"]["f1"]),
        "control_shuffled_minus_champion_f1": float(result["control_shuffled"]["f1"] - result["champion"]["f1"]),
        "candidate_real_minus_control_shuffled_f1": float(result["candidate_real"]["f1"] - result["control_shuffled"]["f1"]),
    }
    return result


def _primary_view(predictions: dict[int, dict[str, Any]], y_all: np.ndarray) -> dict[str, Any]:
    segments = []
    for origin, low, high in ((67, 68, 75), (75, 76, 79)):
        prediction = predictions[origin]
        mask = ((prediction["score_cycles"] >= low) & (prediction["score_cycles"] <= high))
        rows = prediction["score_rows"][mask].astype(np.int64)
        segments.append({
            "origin": origin, "mask": mask, "rows": rows,
            "ids": prediction["score_record_ids"][mask].astype(str),
            "cycles": prediction["score_cycles"][mask].astype(np.int64),
            "y": y_all[rows].astype(np.int64),
            "p_champion": prediction["p_champion"][mask].astype(np.float64),
            "p_candidate_real": prediction["p_candidate_real"][mask].astype(np.float64),
            "p_control_shuffled": prediction["p_control_shuffled"][mask].astype(np.float64),
            "thresholds": prediction["decision_thresholds"][mask].astype(np.float64),
            "present": prediction["present_mask"][mask].astype(bool),
            "entry_friction": prediction["entry_friction"][mask].astype(str),
            "text_load_group": prediction["text_load_group"][mask].astype(str),
        })
    ids = np.concatenate([segment["ids"] for segment in segments])
    if len(ids) != 20_903 or len(set(ids.tolist())) != 20_903:
        raise PilotError("Primary unique evaluation rows are not 20,903 unique source records")
    result: dict[str, Any] = {}
    for key in ("ids", "cycles", "y", "p_champion", "p_candidate_real", "p_control_shuffled",
                "thresholds", "present", "entry_friction", "text_load_group"):
        result[key] = np.concatenate([segment[key] for segment in segments])
    result["segments"] = segments
    return result


def _subgroup_metrics(primary: dict[str, Any]) -> dict[str, Any]:
    n = len(primary["y"])
    group_values = {
        "availability": np.where(primary["present"], "present", "missing"),
        "entry_friction": primary["entry_friction"],
        "inherited_text_load_group": primary["text_load_group"],
    }
    groups: dict[str, Any] = {}
    all_specs = [("overall", np.ones(n, dtype=bool))]
    for dimension, values in group_values.items():
        for value in sorted(set(str(item) for item in values)):
            all_specs.append((f"{dimension}={value}", np.asarray(values).astype(str) == value))
    for name, mask in all_specs:
        y = primary["y"][mask]
        support = {"n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum())}
        supported = support["n"] >= 200 and support["positive"] >= 30 and support["negative"] >= 30
        metrics: dict[str, Any] | None = None
        harm_f1 = harm_recall = None
        if len(y):
            probability_keys = ("p_champion", "p_candidate_real", "p_control_shuffled")
            metrics = {}
            threshold = primary["thresholds"][mask]
            for key in probability_keys:
                p = primary[key][mask]
                metrics[key] = binary_metrics(y, p, (p >= threshold).astype(np.int64))
            harm_f1 = max(0.0, metrics["p_champion"]["f1"] - metrics["p_candidate_real"]["f1"])
            harm_recall = max(0.0, metrics["p_champion"]["recall"] - metrics["p_candidate_real"]["recall"])
        groups[name] = {"support": support, "supported_for_gate": bool(supported),
                        "metrics": metrics, "candidate_f1_harm": harm_f1,
                        "candidate_positive_recall_harm": harm_recall}
    supported_groups = [item for item in groups.values() if item["supported_for_gate"]]
    worst_f1_harm = max((float(item["candidate_f1_harm"]) for item in supported_groups), default=None)
    worst_recall_harm = max((float(item["candidate_positive_recall_harm"])
                             for item in supported_groups), default=None)
    return {"dimensions": list(group_values), "groups": groups,
            "supported_group_count": len(supported_groups),
            "worst_supported_candidate_f1_harm": worst_f1_harm,
            "worst_supported_candidate_positive_recall_harm": worst_recall_harm,
            "gate_has_supported_groups": bool(supported_groups)}


def _metric_view_from_arrays(name: str, y: np.ndarray, p_champion: np.ndarray,
                             p_candidate: np.ndarray, p_control: np.ndarray,
                             thresholds: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    mask = np.ones(len(y), dtype=bool)
    pred = {"p_champion": p_champion, "p_candidate_real": p_candidate,
            "p_control_shuffled": p_control, "decision_thresholds": thresholds}
    metrics = _metric_bundle(y, pred, mask)
    return {"name": name, "rows": int(len(y)),
            "cycle_min": int(cycles.min()) if len(cycles) else None,
            "cycle_max": int(cycles.max()) if len(cycles) else None,
            "metrics": metrics}


def _build_windows(predictions: dict[int, dict[str, Any]], y_all: np.ndarray) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    specs = (("E67_W4", 67, 68, 71), ("E67_W5", 67, 68, 72),
             ("E67_W6", 67, 68, 73), ("E67_H12", 67, 68, 79),
             ("E75_W6", 75, 76, 79))
    for name, origin, low, high in specs:
        prediction = predictions[origin]
        mask = ((prediction["score_cycles"] >= low) & (prediction["score_cycles"] <= high))
        rows = prediction["score_rows"][mask].astype(np.int64)
        windows[name] = _metric_view_from_arrays(
            name, y_all[rows].astype(np.int64), prediction["p_champion"][mask],
            prediction["p_candidate_real"][mask], prediction["p_control_shuffled"][mask],
            prediction["decision_thresholds"][mask], prediction["score_cycles"][mask])
    primary = _primary_view(predictions, y_all)
    windows["PRIMARY_UNIQUE"] = _metric_view_from_arrays(
        "PRIMARY_UNIQUE", primary["y"], primary["p_champion"],
        primary["p_candidate_real"], primary["p_control_shuffled"],
        primary["thresholds"], primary["cycles"])
    return windows, primary


def _compute_gates(windows: dict[str, Any], subgroups: dict[str, Any],
                   prelabel_checks: dict[str, Any]) -> dict[str, Any]:
    primary = windows["PRIMARY_UNIQUE"]["metrics"]
    e67_h12 = windows["E67_H12"]["metrics"]
    e75_w6 = windows["E75_W6"]["metrics"]
    primary_gain = primary["gains"]["candidate_real_minus_champion_f1"]
    e67_gain = e67_h12["gains"]["candidate_real_minus_champion_f1"]
    e75_gain = e75_w6["gains"]["candidate_real_minus_champion_f1"]
    candidate_control = primary["gains"]["candidate_real_minus_control_shuffled_f1"]
    e75_control_gain = e75_w6["gains"]["control_shuffled_minus_champion_f1"]
    worst_f1 = subgroups["worst_supported_candidate_f1_harm"]
    worst_recall = subgroups["worst_supported_candidate_positive_recall_harm"]
    qa_ok = all(item.get("missing_route_exact") is True for item in prelabel_checks.values()
                if isinstance(item, dict) and "missing_route_exact" in item)
    qa_ok = qa_ok and subgroups["gate_has_supported_groups"]
    scientific_checks = {
        "primary_gain_vs_champion_ge_0_0005": primary_gain >= 0.0005,
        "e67_h12_gain_nonnegative": e67_gain >= 0.0,
        "e75_w6_gain_nonnegative": e75_gain >= 0.0,
        "primary_candidate_minus_control_ge_0_0002": candidate_control >= 0.0002,
        "e75_control_gain_nonnegative": e75_control_gain >= 0.0,
        "supported_subgroup_f1_harm_le_0_01": worst_f1 is not None and worst_f1 <= 0.01,
        "supported_subgroup_positive_recall_harm_le_0_02": worst_recall is not None and worst_recall <= 0.02,
        "prelabel_qa_and_missing_route_exact": qa_ok,
    }
    exploratory_formula = ((primary_gain > 0 and e75_gain >= -0.0005)
                           or (e75_gain >= 0.0005 and primary_gain >= -0.0005))
    exploratory_checks = {
        "candidate_gain_pattern": bool(exploratory_formula),
        "supported_subgroup_f1_harm_le_0_02": worst_f1 is not None and worst_f1 <= 0.02,
        "supported_subgroup_positive_recall_harm_le_0_03": worst_recall is not None and worst_recall <= 0.03,
        "prelabel_qa_and_missing_route_exact": qa_ok,
    }
    scientific_pass = all(scientific_checks.values())
    exploratory_pass = all(exploratory_checks.values())
    mechanism_checks = {"primary_candidate_minus_control_ge_0_0002": candidate_control >= 0.0002,
                        "e75_control_gain_nonnegative": e75_control_gain >= 0.0}
    return {
        "scientific": {"status": "PASS" if scientific_pass else "FAIL",
                       "checks": scientific_checks, "primary_gain": primary_gain,
                       "e67_h12_gain": e67_gain, "e75_w6_gain": e75_gain,
                       "candidate_minus_control_primary": candidate_control,
                       "e75_control_gain": e75_control_gain,
                       "worst_supported_subgroup_f1_harm": worst_f1,
                       "worst_supported_subgroup_positive_recall_harm": worst_recall},
        "exploratory": {"status": "ELIGIBLE" if exploratory_pass else "NOT_ELIGIBLE",
                        "checks": exploratory_checks},
        "mechanism": {"status": "SEPARATED_FROM_SHUFFLED_CONTROL" if all(mechanism_checks.values())
                      else "NOT_SEPARATED_FROM_SHUFFLED_CONTROL",
                      "checks": mechanism_checks,
                      "interpretation": "A separation supports the whole real-context pipeline; shuffled geometry and HGB axis sensitivity prevent an isolated causal token-order claim."},
        "overall": "SCIENTIFIC_PASS" if scientific_pass else
                   "EXPLORATORY_ONLY_ELIGIBLE" if exploratory_pass else "NO_ADVANCE",
    }


def _report_markdown(results: dict[str, Any]) -> str:
    windows = results["windows"]
    lines = ["# CLS_PPMI_CONTEXT_V1 pilot results", "",
             f"Status: **{results['gates']['overall']}**",
             "",
             "The pilot used four prefix-specific PPMI/SVD representations and four fixed HGB predictor fits. "
             "Evaluation labels were opened only after the immutable prelabel commit.", "",
             "## Main windows", "",
             "| View | n | Champion F1 | Real candidate F1 | Shuffled control F1 | Candidate gain | Candidate minus control |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("E67_W4", "E67_W5", "E67_W6", "E67_H12", "E75_W6", "PRIMARY_UNIQUE"):
        item = windows[name]
        metrics = item["metrics"]
        gains = metrics["gains"]
        lines.append(f"| {name} cycles {item['cycle_min']}-{item['cycle_max']} | {item['rows']} | "
                     f"{metrics['champion']['f1']:.6f} | {metrics['candidate_real']['f1']:.6f} | "
                     f"{metrics['control_shuffled']['f1']:.6f} | "
                     f"{gains['candidate_real_minus_champion_f1']:+.6f} | "
                     f"{gains['candidate_real_minus_control_shuffled_f1']:+.6f} |")
    lines.extend(["", "## Gates", "",
                  f"- Scientific gate: **{results['gates']['scientific']['status']}**.",
                  f"- Exploratory gate: **{results['gates']['exploratory']['status']}**.",
                  f"- Mechanism comparison: **{results['gates']['mechanism']['status']}**.",
                  f"- Worst supported subgroup F1 harm: {results['subgroups']['worst_supported_candidate_f1_harm']}.",
                  f"- Worst supported subgroup positive recall harm: {results['subgroups']['worst_supported_candidate_positive_recall_harm']}.",
                  "", "## Readout", "",
                  results["interpretation"], "",
                  "The real-versus-shuffled comparison changes the learned context geometry and SVD axes as well as local order. "
                  "It is evidence about the full representation pipeline, not an isolated causal order effect.", "",
                  "## Boundary and next step", "",
                  f"- Official source SHA-256: `{results['official_source']['sha256']}`.",
                  f"- Query rows scored: {results['evaluation']['unique_primary_rows']} unique rows; target column read once after the committed prelabel.",
                  f"- Recommendation: {results['recommended_next_step']}", ""])
    return "\n".join(lines)


def command_evaluate() -> None:
    freeze = _verify_freeze(require_committed=True)
    prelabel_check = _verify_prelabel_artifacts(freeze, require_committed=True)
    if RESULTS_PATH.exists() or REPORT_PATH.exists() or POSTLABEL_VERIFICATION_PATH.exists():
        raise PilotError("Evaluation outputs already exist; the one-score boundary forbids a second run")
    source_root = Path(freeze["source_root"])
    train_path = source_root / TRAIN_REL
    if sha256_file(train_path) != TRAIN_SHA256:
        raise PilotError("Official Train source changed immediately before label read")
    # This is the first query-target read in the campaign, guarded by the committed prelabel above.
    targets = pd.read_csv(train_path, usecols=["artifact_signal"])["artifact_signal"]
    y_all = pd.to_numeric(targets, errors="raise").to_numpy(dtype=np.int64)
    if len(y_all) != 50_608 or not np.isin(y_all, (0, 1)).all():
        raise PilotError("Official artifact_signal labels are not aligned binary 0/1 values")
    predictions = prelabel_check["predictions"]
    windows, primary = _build_windows(predictions, y_all)
    subgroups = _subgroup_metrics(primary)
    gates = _compute_gates(windows, subgroups, prelabel_check["checks"])
    cycle_signs: dict[str, Any] = {}
    for cycle in sorted(set(primary["cycles"].tolist())):
        mask = primary["cycles"] == cycle
        metrics = _metric_bundle(primary["y"][mask], {
            "p_champion": primary["p_champion"][mask],
            "p_candidate_real": primary["p_candidate_real"][mask],
            "p_control_shuffled": primary["p_control_shuffled"][mask],
            "decision_thresholds": primary["thresholds"][mask],
        }, np.ones(int(mask.sum()), dtype=bool))
        cycle_signs[str(int(cycle))] = {
            "rows": int(mask.sum()),
            "candidate_minus_champion_f1": metrics["gains"]["candidate_real_minus_champion_f1"],
            "candidate_minus_control_f1": metrics["gains"]["candidate_real_minus_control_shuffled_f1"],
            "control_minus_champion_f1": metrics["gains"]["control_shuffled_minus_champion_f1"],
        }
    source_counts = {str(label): int(np.count_nonzero(y_all == label)) for label in (0, 1)}
    if gates["overall"] == "SCIENTIFIC_PASS":
        recommendation = "Root should review whether the positive PPMI result warrants one distinct, source-frozen CLS mechanism; keep this pilot closed and do not add fits before that review."
    elif gates["overall"] == "EXPLORATORY_ONLY_ELIGIBLE":
        recommendation = "Root should review the exploratory-only result and authorize at most one distinct source-only novelty scan before any new fit."
    else:
        recommendation = "Close this PPMI-context family; root should select one distinct, source-only novelty scan from the untested CLS mechanisms before any new fit."
    primary_gain = windows["PRIMARY_UNIQUE"]["metrics"]["gains"]["candidate_real_minus_champion_f1"]
    results = {
        "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
        "freeze_signature": freeze["freeze_signature"],
        "prelabel_signature": prelabel_check["prelabel"]["prelabel_signature"],
        "evaluation": {"target_column": "artifact_signal", "target_column_read_count": 1,
                       "query_targets_opened_after_committed_prelabel": True,
                       "query_rows_opened": int(len(set(np.concatenate([
                           item["score_rows"].astype(np.int64) for item in predictions.values()])))),
                       "unique_primary_rows": int(len(primary["y"])),
                       "unique_primary_positive": int(primary["y"].sum()),
                       "unique_primary_negative": int(len(primary["y"]) - primary["y"].sum()),
                       "full_train_label_counts": source_counts,
                       "one_logical_evaluation": True},
        "official_source": freeze["official_source"],
        "fit_label_counts": read_json(FIT_SUMMARY_PATH)["fit_label_counts"],
        "windows": windows,
        "cycle_signs_primary_unique": cycle_signs,
        "subgroups": subgroups,
        "gates": gates,
        "prelabel_checks": prelabel_check["checks"],
        "primary_f1_gain": primary_gain,
        "interpretation": ("The fitted real-order PPMI/SVD+HGB policy was compared with the full frozen champion and a same-population, same-estimator within-document shuffled-context control."),
        "recommended_next_step": recommendation,
        "query_targets_read": True, "test_or_sample_opened": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(RESULTS_PATH, results, write_once=True)
    report = _report_markdown(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        raise PilotError(f"Write-once evaluation report already exists: {REPORT_PATH}")
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    verification = {
        "schema_version": 1, "pilot_id": "CLS_PPMI_CONTEXT_V1",
        "status": "PASS", "freeze_signature": freeze["freeze_signature"],
        "prelabel_signature": prelabel_check["prelabel"]["prelabel_signature"],
        "results_sha256": sha256_file(RESULTS_PATH), "report_sha256": sha256_file(REPORT_PATH),
        "target_column_read_count": 1, "query_targets_opened_after_committed_prelabel": True,
        "independent_metric_checks": "F1, precision, recall, confusion and Brier readback in the evaluation pass",
        "gates_recomputed_from_saved_prelabel": True,
        "query_targets_read": True, "test_or_sample_opened": False,
    }
    atomic_json(POSTLABEL_VERIFICATION_PATH, verification, write_once=True)
    print(json.dumps({"status": gates["overall"], "results": str(RESULTS_PATH),
                      "report": str(REPORT_PATH), "primary_f1_gain": primary_gain,
                      "scientific_gate": gates["scientific"]["status"],
                      "exploratory_gate": gates["exploratory"]["status"],
                      "mechanism_gate": gates["mechanism"]["status"],
                      "query_rows": results["evaluation"]["unique_primary_rows"],
                      "query_targets_read": True}, sort_keys=True))


def verify_artifacts(phase: str = "prelabel", *, require_committed: bool = True) -> dict[str, Any]:
    freeze = _verify_freeze(require_committed=require_committed)
    prelabel_check = _verify_prelabel_artifacts(freeze, require_committed=require_committed)
    if phase == "prelabel":
        verification = read_json(PRELABEL_VERIFICATION_PATH)
        if verification.get("status") != "PASS" or verification.get("query_targets_read") is not False:
            raise PilotError("Prelabel verification receipt fails its boundary")
        if require_committed:
            assert_committed_exact(ROOT, PRELABEL_VERIFICATION_PATH,
                                   PRELABEL_VERIFICATION_PATH.relative_to(ROOT).as_posix())
        return {"status": "PASS", "phase": phase,
                "prelabel_signature": prelabel_check["prelabel"]["prelabel_signature"],
                "checks": prelabel_check["checks"], "query_targets_read": False}
    if phase != "postlabel":
        raise PilotError("phase must be prelabel or postlabel")
    if not RESULTS_PATH.is_file() or not REPORT_PATH.is_file() or not POSTLABEL_VERIFICATION_PATH.is_file():
        raise PilotError("Postlabel results/report/verifier receipt missing")
    results = read_json(RESULTS_PATH)
    verification = read_json(POSTLABEL_VERIFICATION_PATH)
    if results.get("prelabel_signature") != prelabel_check["prelabel"]["prelabel_signature"] \
            or results.get("query_targets_read") is not True \
            or results.get("evaluation", {}).get("target_column_read_count") != 1:
        raise PilotError("Postlabel result boundary or prelabel binding is invalid")
    if verification.get("results_sha256") != sha256_file(RESULTS_PATH) \
            or verification.get("report_sha256") != sha256_file(REPORT_PATH):
        raise PilotError("Postlabel result/report hashes differ from verification receipt")
    for window in results["windows"].values():
        for key in ("champion", "candidate_real", "control_shuffled"):
            metric = window["metrics"][key]
            if not np.isclose(metric["f1"], metric["independent_f1"], rtol=0.0, atol=1e-15):
                raise PilotError("Independent F1 readback differs in saved result")
    if require_committed:
        for path in (RESULTS_PATH, REPORT_PATH, POSTLABEL_VERIFICATION_PATH):
            assert_committed_exact(ROOT, path, path.relative_to(ROOT).as_posix())
    return {"status": "PASS", "phase": phase,
            "overall": results["gates"]["overall"],
            "primary_f1_gain": results["primary_f1_gain"],
            "results_sha256": sha256_file(RESULTS_PATH),
            "query_targets_read": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic", help="Run isolated synthetic contract QA")
    freeze_parser = sub.add_parser("freeze", help="Create the write-once source/input freeze")
    freeze_parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    freeze_parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    sub.add_parser("preflight", help="Estimate frozen four-fit resource needs without reading labels")
    fit_parser = sub.add_parser("fit", help="Run exactly four frozen representation and HGB starts")
    fit_parser.add_argument("--lock-wait-seconds", type=int, default=60)
    sub.add_parser("evaluate", help="Open query labels once after committed prelabel outputs")
    verify_parser = sub.add_parser("verify", help="Verify committed prelabel or postlabel artifacts")
    verify_parser.add_argument("--phase", choices=("prelabel", "postlabel"), default="prelabel")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            command_synthetic()
        elif args.command == "freeze":
            command_freeze(args.source_root, args.audit_root)
        elif args.command == "preflight":
            command_preflight()
        elif args.command == "fit":
            command_fit(args.lock_wait_seconds)
        elif args.command == "evaluate":
            command_evaluate()
        elif args.command == "verify":
            print(json.dumps(verify_artifacts(args.phase), sort_keys=True))
        else:
            parser.error("unhandled command")
        return 0
    except LockBusy as exc:
        print(json.dumps({"status": "WAITING_FOR_SHARED_LOCK", "reason": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 75
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
