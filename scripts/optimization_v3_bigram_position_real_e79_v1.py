"""Execute one authorized full-Train REAL bigram-position E79 benchmark.

Stages: synthetic -> preflight -> reserve -> fit -> freeze-trained -> infer.
Only the last stage opens official Test features; Test labels and Kaggle are
never accessed. The old pilot and other E79 benchmarks remain immutable.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom  # noqa: E402
from scripts import optimization_v3_positional_bag_offset_v1 as offset  # noqa: E402
from src.models.optimization_v3_bigram_position_pilot_v1 import (  # noqa: E402
    reconstruct_original_pos_logit,
    route_candidate,
    zero_gamma_equivalence,
)
from src.models.optimization_v3_bigram_position_readiness_v1 import (  # noqa: E402
    build_bigram_blocks,
)
from src.models.positional_bag_offset_v1 import fit_offset_coefficients  # noqa: E402


CARD_ID = "CLS_BIGRAM_POSITION_REAL_E79_V1"
STAGE = "REAL_E79_FULLTRAIN_GAMMA"
FIT_ID = f"{CARD_ID}_GAMMA"
AMENDMENT_ID = "CLS_BIGRAM_POSITION_REAL_E79_V1_CAP150_2026_11_04"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
TEST_SHA = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
TRAIN_ROWS = 50_608
TEST_ROWS = 29_421
THREADS = 2
RSS_CEILING = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
THRESHOLD = 0.50
TOL = 1e-12

REPORT = ROOT / "reports/optimization_v3/bigram_position_real_e79_v1"
RUN = ROOT / "experiments/runs/optimization_v3_research_next_v1/BIGRAM_POSITION_REAL_E79_V1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
TEST = ROOT / "data/raw/classification/Test.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
TRAIN_FREEZE = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1/E79_TRAINED_FREEZE.json"
BATCH_MANIFEST = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1/BATCH_MANIFEST.json"
TEST_BUNDLE = cgeom.OUTPUT_NPZ
TEST_INFERENCE_RECEIPT = cgeom.RUN_ROOT / "inference_receipt.json"
CSV = ROOT / "submissions/optimization_v3/bigram_position_real_e79_v1/classification_bigram_position_real_e79.csv"

PREFIT_PATH = REPORT / "PRE_FIT_FREEZE.json"
SYNTHETIC_PATH = REPORT / "SYNTHETIC_QA.json"
RESERVATION_PATH = REPORT / "RESERVATION.json"
FIT_RECEIPT_PATH = REPORT / "FIT_RECEIPT.json"
TRAINED_FREEZE_PATH = REPORT / "TRAINED_STATE_FREEZE.json"
INFERENCE_QA_PATH = REPORT / "INFERENCE_QA.json"
MANIFEST_PATH = REPORT / "BENCHMARK_MANIFEST.json"
ACCOUNTING_PATH = REPORT / "BUDGET_ACCOUNTING.json"
RESULTS_PATH = REPORT / "RESULTS.md"

MATRIX_PATH = RUN / "real_train_bigram.npz"
TRAIN_ARRAYS_PATH = RUN / "train_offset_and_rows.npz"
GAMMA_PATH = RUN / "fits" / f"{FIT_ID}.npz"
RUN_RECEIPT_PATH = RUN / "receipts" / f"{FIT_ID}.json"

SOURCE_PATHS = (
    "scripts/optimization_v3_bigram_position_real_e79_v1.py",
    "scripts/optimization_v3_a02_cgeom_benchmark_v1.py",
    "scripts/optimization_v3_bigram_position_pilot_v1.py",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "src/models/optimization_v3_bigram_position_pilot_v1.py",
    "src/models/optimization_v3_bigram_position_readiness_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "reports/optimization_v3/bigram_position_real_e79_v1/RECIPE.json",
    "reports/optimization_v3/bigram_position_real_e79_v1/BENCHMARK_AUTHORIZATION.md",
)

REUSED_UNITS = (
    "A02_CGEOM_BENCHMARK_V1_E79_C2B_STRUCTURED",
    "A02_CGEOM_BENCHMARK_V1_E79_C2B_SCALER",
    "A02_CGEOM_BENCHMARK_V1_E79_C2B_OHE",
    "A02_CGEOM_BENCHMARK_V1_E79_C2B_LR_INTEGRATION_R1",
    "A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1",
    "A02_CGEOM_BENCHMARK_V1_E79_C1B_STRUCTURED_INTEGRATION_R1",
    "A02_CGEOM_BENCHMARK_V1_E79_C1B_HGB_INTEGRATION_R1",
)


class BenchmarkError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected JSON object: {path}")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise BenchmarkError(f"Refusing to overwrite a different immutable output: {path}")
        return payload
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with temp.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return payload


def _write_npz_once(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                raise BenchmarkError(f"Existing array bundle schema differs: {path}")
            for key, value in arrays.items():
                if not np.array_equal(existing[key], np.asarray(value)):
                    raise BenchmarkError(f"Existing array bundle values differ for {key}: {path}")
        return
    temp = path.with_name(path.name + f".{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(temp, **arrays)
        with temp.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise BenchmarkError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _head() -> str:
    return _git("rev-parse", "HEAD")


def _require_clean_tree() -> None:
    if _git("status", "--porcelain"):
        raise BenchmarkError("Working tree must be committed before this stage")


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise BenchmarkError(f"Required source is missing: {relative}")
        result[relative] = _sha_file(path)
    return result


def _resource_snapshot(*, initial: bool) -> dict[str, Any]:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    if rss > RSS_CEILING:
        raise BenchmarkError(f"Current process RSS exceeds 6 GiB: {rss}")
    floor = INITIAL_FREE_MIN if initial else FREE_RESERVE
    if available < floor:
        raise BenchmarkError(f"Available RAM {available} is below required {floor}")
    busy: list[dict[str, Any]] = []
    for other in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if other.pid == os.getpid():
                continue
            name = (other.info.get("name") or "").lower()
            command = " ".join(other.info.get("cmdline") or []).lower()
            if "python" in name and any(token in command for token in (
                    "optimization_v3_", "positional_", "bigram_position")):
                busy.append({"pid": other.pid, "name": name, "command": command[:200]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if busy:
        raise BenchmarkError(f"Another project research Python job is active: {busy}")
    return {"process_rss_bytes": rss, "available_ram_bytes": available,
            "required_free_bytes": floor, "jobs": 1, "threads": THREADS,
            "rss_ceiling_bytes": RSS_CEILING, "other_research_jobs": busy}


def _train_feature_frame() -> pd.DataFrame:
    if _sha_file(TRAIN) != TRAIN_SHA:
        raise BenchmarkError("Official Classification Train hash differs from the frozen recipe")
    columns = ["record_id", "cycle_code", *cgeom.EXPANDED_FEATURES, "signal_stream"]
    frame = pd.read_csv(TRAIN, usecols=columns, dtype={"record_id": str})
    _validate_identity_features(frame, train=True)
    if "artifact_signal" in frame.columns:
        raise BenchmarkError("Target unexpectedly entered the target-free preflight frame")
    return frame


def _validate_identity_features(frame: pd.DataFrame, *, train: bool) -> None:
    expected_rows = TRAIN_ROWS if train else TEST_ROWS
    if len(frame) != expected_rows or frame.columns.has_duplicates:
        raise BenchmarkError(f"Unexpected row count or duplicate feature columns: {len(frame)}")
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != expected_rows or np.any(ids == ""):
        raise BenchmarkError("Official row IDs are empty or duplicated")
    if train:
        if cycles.min() != 47 or cycles.max() != 79 or not np.array_equal(np.unique(cycles), np.arange(47, 80)):
            raise BenchmarkError("Train cycle support is not the complete 47-79 range")
    else:
        if cycles.min() != 80 or cycles.max() != 91 or not np.array_equal(np.unique(cycles), np.arange(80, 92)):
            raise BenchmarkError("Test cycle support is not the complete 80-91 range")


def _ledger_state(path: Path = LEDGER) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    events = offset._ledger_events(path)
    return events, offset.ledger_snapshot(events), _sha_file(path)


def _assert_expected_reused_units() -> dict[str, Any]:
    if _sha_file(TRAIN_FREEZE) != "ab0307e6e1c9778ed0f091a3d6ed55e0ac8705ae31b9cb29b9f083e46b5c2b82":
        raise BenchmarkError("A02-CGEOM E79 trained-state freeze changed")
    trained = _json(TRAIN_FREEZE)
    batch = _json(BATCH_MANIFEST)
    if (trained.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or trained.get("train_rows") != TRAIN_ROWS
            or trained.get("train_cycles") != [47, 79]
            or trained.get("official_inputs_sha256", {}).get("data/raw/classification/Train.csv") != TRAIN_SHA
            or batch.get("trained_freeze_sha256") != _sha_file(TRAIN_FREEZE)
            or not isinstance(batch.get("prediction_bundle_sha256"), str)
            or len(batch["prediction_bundle_sha256"]) != 64):
        raise BenchmarkError("A02-CGEOM full-Train freeze/manifest lineage does not match")
    if batch.get("test_target_loaded") is not False or batch.get("used_test_for_model_selection") is not False:
        raise BenchmarkError("A02-CGEOM lineage does not attest to Test-label isolation")

    # Verify the reusable A02 text model and its Train-only receipt without
    # opening its saved Test prediction bundle. Test artifacts are inspected
    # only by infer(), after this benchmark's trained-state freeze is committed.
    a02_model_sha = cgeom.sha256_file(cgeom.A02_MODEL)
    a02_receipt = _json(cgeom.A02_FIT_RECEIPT)
    a02_receipt_body = dict(a02_receipt)
    a02_receipt_sha = a02_receipt_body.pop("receipt_sha256", None)
    if (a02_model_sha != _json(cgeom.A02_MANIFEST).get("model_sha256")
            or a02_receipt.get("fit_id") != "A02_CONT_E79_OBSERVED_FULLTRAIN"
            or a02_receipt_sha != cgeom.canonical_sha(a02_receipt_body)
            or a02_receipt.get("receipt_sha256") != _json(cgeom.A02_MANIFEST).get("fit_receipt_sha256")
            or a02_receipt.get("model_sha256") != a02_model_sha
            or a02_receipt.get("text_prefix_rows") != TRAIN_ROWS
            or a02_receipt.get("test_data_used") is not False
            or a02_receipt.get("query_labels_passed_to_fit") is not False
            or _json(cgeom.A02_STATE) != cgeom.joblib.load(cgeom.A02_MODEL).audit_state()):
        raise BenchmarkError("A02 full-Train text model/receipt/state lineage does not match")
    a02_model = cgeom.joblib.load(cgeom.A02_MODEL)
    if a02_model.text_space.fit_rows != TRAIN_ROWS or a02_model.text_space.origin_cycle != 79:
        raise BenchmarkError("A02 text vectorizer was not fit on the complete official Train population")

    unit_records: dict[str, Any] = {}
    for fit_id in REUSED_UNITS:
        receipt_path = cgeom.UNIT_ROOT / f"{fit_id}.json"
        artifact_path = cgeom.UNIT_ROOT / f"{fit_id}.joblib"
        receipt = _json(receipt_path)
        verified_receipt = cgeom._verify_receipt(fit_id)
        artifact_sha = _sha_file(artifact_path)
        if (receipt.get("fit_id") != fit_id
                or verified_receipt.get("artifact_sha256") != artifact_sha
                or receipt.get("artifact_sha256") != artifact_sha
                or trained.get("fitted_state_sha256", {}).get(fit_id) != artifact_sha
                or receipt.get("fit_population_rows") != TRAIN_ROWS
                or receipt.get("fit_cycle_min_max") != [47, 79]
                or receipt.get("train_sha256") != TRAIN_SHA):
            raise BenchmarkError(f"Full-Train receipt or fitted artifact mismatch: {fit_id}")
        unit_records[fit_id] = {
            "receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": _sha_file(receipt_path),
            "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
            "artifact_sha256": artifact_sha,
            "fit_rows": receipt["fit_population_rows"],
            "fit_cycles": receipt["fit_cycle_min_max"],
            "component": receipt.get("component"),
            "receipt_verified_by_cgeom": True,
        }
    if len(unit_records) != 7:
        raise BenchmarkError("Not all required frozen C2_B, original POS, and C1_B states were audited")
    return {
        "trained_freeze_sha256": _sha_file(TRAIN_FREEZE),
        "batch_manifest_sha256": _sha_file(BATCH_MANIFEST),
        "a02_model_sha256": a02_model_sha,
        "a02_model_state_sha256": cgeom.canonical_sha(a02_model.audit_state()),
        "a02_fit_receipt_sha256": _sha_file(cgeom.A02_FIT_RECEIPT),
        "units": unit_records,
        "test_bundle_path": TEST_BUNDLE.relative_to(ROOT).as_posix(),
        "test_bundle_sha256_from_frozen_manifest": batch["prediction_bundle_sha256"],
        "test_bundle_opened_or_hashed_by_this_preflight": False,
    }


def _make_real_train_cache(frame: pd.DataFrame, vectorizer: Any) -> dict[str, Any]:
    docs, text_missing = cgeom.positional._text_values(frame["signal_stream"])
    blocks, metadata = build_bigram_blocks(docs, vectorizer)
    real = sparse.csr_matrix(blocks["real"], dtype=np.float64)
    no_known = np.asarray(metadata["known_occurrence_count"] == 0, dtype=bool)
    present = ~np.asarray(text_missing, dtype=bool)
    if real.shape[0] != len(frame) or no_known.shape != (len(frame),) or present.shape != (len(frame),):
        raise BenchmarkError("Full-Train REAL bigram matrix/availability rows do not align")
    if not np.isfinite(real.data).all() or not metadata["pair_bag_preserved"].all() or not metadata["segment_sizes_preserved"].all():
        raise BenchmarkError("Bigram builder produced invalid features or failed its pair-order control invariants")
    _save_sparse_once(MATRIX_PATH, real)
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    npz = {
        "record_ids": ids,
        "cycles": cycles,
        "present_mask": present,
        "no_known_bigram": no_known,
    }
    return {
        "matrix": real,
        "metadata": metadata,
        "rows": {
            "record_ids": ids,
            "cycles": cycles,
            "present_mask": present,
            "no_known_bigram": no_known,
        },
        "matrix_file_sha256": _sha_file(MATRIX_PATH),
        "matrix_logical_sha256": offset.sha256_csr(real),
        "matrix_shape": list(real.shape),
        "matrix_nnz": int(real.nnz),
        "bigram_vocabulary_size": int(metadata["bigram_vocabulary_size"]),
        "nonzero_rows": int(np.count_nonzero(metadata["known_occurrence_count"])),
        "known_occurrences": int(metadata["known_occurrence_count"].sum()),
        "row_array_hashes": {key: offset.sha256_array(value) for key, value in npz.items()},
        "test_or_sample_opened": False,
    }


def _save_sparse_once(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logical = offset.sha256_csr(matrix)
    if path.exists():
        loaded = sparse.load_npz(path).tocsr()
        if offset.sha256_csr(loaded) != logical:
            raise BenchmarkError(f"Existing immutable sparse cache differs: {path}")
        return
    temp = path.with_name(path.name + f".{os.getpid()}.partial.npz")
    try:
        sparse.save_npz(temp, sparse.csr_matrix(matrix, dtype=np.float64))
        with temp.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _verify_position_offset(frame: pd.DataFrame, vectorizer: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base, unigram_position, audit = cgeom._c2b_matrices(frame, vectorizer=vectorizer, train_side=True)
    if unigram_position is None:
        raise BenchmarkError("Original unigram position matrix was not produced")
    c2b_fit_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[3]
    pos_fit_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[6]
    c2b_receipt = _json(cgeom.UNIT_ROOT / f"{c2b_fit_id}.json")
    pos_receipt = _json(cgeom.UNIT_ROOT / f"{pos_fit_id}.json")
    c2b_state = cgeom._load_unit(c2b_fit_id)
    pos_state = cgeom._load_unit(pos_fit_id)
    bag = c2b_state["estimator"]
    original_gamma = np.asarray(pos_state["gamma"], dtype=np.float64).reshape(-1)
    z0 = np.asarray(bag.decision_function(base), dtype=np.float64).reshape(-1)
    base_sha = offset.sha256_csr(base)
    position_sha = offset.sha256_csr(unigram_position)
    if (base_sha != c2b_receipt.get("fit_matrix_sha256")
            or base_sha != pos_receipt.get("base_matrix_sha256")
            or position_sha != pos_receipt.get("fit_matrix_sha256")
            or offset.sha256_array(z0) != pos_receipt.get("z0_sha256")
            or offset.sha256_array(bag.coef_) != pos_receipt.get("bag_lr_coefficient_sha256")
            or cgeom.canonical_sha(cgeom._state_summary(pos_state)) != pos_receipt.get("state_sha256")):
        raise BenchmarkError("Recomputed full-Train C2_B/original POS feature lineage differs from its fitted receipts")
    _z_bag, z_original = reconstruct_original_pos_logit(
        base, bag.coef_[0], bag.intercept_, unigram_position, original_gamma
    )
    pos_probability = np.asarray(expit(z_original), dtype=np.float64)
    if not np.isfinite(pos_probability).all():
        raise BenchmarkError("Reconstructed original POS full-Train probabilities are non-finite")
    return z_original, pos_probability, {
        "base_matrix_sha256": base_sha,
        "unigram_position_matrix_sha256": position_sha,
        "c2b_z0_sha256": offset.sha256_array(z0),
        "original_gamma_sha256": offset.sha256_array(original_gamma),
        "original_pos_state_sha256": pos_receipt["state_sha256"],
        "z_original_sha256": offset.sha256_array(z_original),
        "pos_probability_sha256": offset.sha256_array(pos_probability),
        "base_shape": list(base.shape),
        "unigram_position_shape": list(unigram_position.shape),
        "a02_vectorizer_compatibility": cgeom.audit_c2b_vectorizer_compatibility(),
        "matrix_audit": audit,
    }


def synthetic() -> dict[str, Any]:
    """Synthetic builder, one synthetic fit, persistence/reload, and route QA."""
    docs = [
        "alpha beta gamma", "alpha beta delta", "alpha beta gamma",
        "beta gamma alpha", "beta gamma delta", "delta alpha beta",
        "gamma alpha beta", "delta beta alpha", "alpha delta gamma",
        "gamma delta beta", "beta alpha delta", "delta gamma alpha",
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0,
                                 strip_accents="unicode", lowercase=True,
                                 token_pattern=r"(?u)\b\w\w+\b")
    vectorizer.fit(docs)
    blocks, metadata = build_bigram_blocks(docs, vectorizer)
    matrix = sparse.csr_matrix(blocks["real"], dtype=np.float64)
    z0 = np.linspace(-0.8, 0.8, len(docs), dtype=np.float64)
    y = np.asarray([0, 1] * (len(docs) // 2), dtype=np.int8)
    with tempfile.TemporaryDirectory(prefix="bigram-real-e79-synthetic-") as tmp:
        root = Path(tmp)
        gamma, optimizer = fit_offset_coefficients(matrix, z0, y, c=1.0)
        gamma_path = root / "gamma.npz"
        receipt_path = root / "receipt.json"
        np.savez_compressed(gamma_path, gamma=gamma)
        receipt = {"fit_id": "SYNTHETIC_ONLY", "matrix_sha256": offset.sha256_csr(matrix),
                   "z_sha256": offset.sha256_array(z0), "y_sha256": offset.sha256_array(y),
                   "gamma_sha256": offset.sha256_array(gamma),
                   "gamma_file_sha256": _sha_file(gamma_path), "optimizer": optimizer}
        receipt["receipt_sha256"] = _sha_value(receipt)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        with np.load(gamma_path, allow_pickle=False) as saved:
            gamma2 = np.asarray(saved["gamma"], dtype=np.float64)
        if (not np.array_equal(gamma, gamma2)
                or _sha_file(gamma_path) != receipt["gamma_file_sha256"]
                or offset.sha256_array(gamma2) != receipt["gamma_sha256"]):
            raise BenchmarkError("Synthetic fitted gamma persistence/reload check failed")

        present = np.asarray([True] * 10 + [False] * 2, dtype=bool)
        no_known = np.asarray(metadata["known_occurrence_count"] == 0, dtype=bool)
        p_a02 = np.linspace(0.15, 0.85, len(docs), dtype=np.float64)
        p_p = np.linspace(0.20, 0.90, len(docs), dtype=np.float64)
        original_pos = np.asarray(expit(z0), dtype=np.float64)
        mix50 = 0.5 * original_pos + 0.5 * p_a02
        mix50[~present] = p_p[~present]
        identity = zero_gamma_equivalence(
            matrix, z0, present_mask=present, p_a02=p_a02, p_p=p_p, mix50_original=mix50
        )
        if not identity["pass"]:
            raise BenchmarkError(f"Synthetic zero-gamma route equivalence failed: {identity}")
        _q, routed = route_candidate(
            matrix, z0, gamma, present_mask=present, no_known_bigram=no_known,
            p_pos_original=original_pos, p_a02=p_a02, p_p=p_p, mix50_original=mix50,
        )
        if (not np.array_equal(routed[~present], p_p[~present])
                or not np.array_equal(routed[present & no_known], mix50[present & no_known])
                or not np.isfinite(routed).all()):
            raise BenchmarkError("Synthetic availability or no-known-bigram route check failed")
        # Resume check: reload the durable gamma/receipt and return it without a fit.
        resumed = np.load(gamma_path, allow_pickle=False)["gamma"]
        if not np.array_equal(resumed, gamma2):
            raise BenchmarkError("Synthetic saved-fit resume check failed")
    result = {
        "status": "PASS_SYNTHETIC_END_TO_END_PERSISTENCE_RESUME",
        "production_fit_started": False,
        "production_test_opened": False,
        "synthetic_rows": len(docs),
        "feature_shape": list(matrix.shape),
        "feature_nnz": int(matrix.nnz),
        "feature_hash": offset.sha256_csr(matrix),
        "pair_order_controls_pass": bool(metadata["pair_bag_preserved"].all()
                                          and metadata["segment_sizes_preserved"].all()),
        "optimizer": optimizer,
        "gamma_persist_reload_bit_exact": True,
        "saved_fit_resume_without_refit": True,
        "zero_gamma_equivalence": identity,
        "availability_route_bit_exact": True,
        "no_known_bigram_fallback_bit_exact": True,
    }
    _write_json_once(SYNTHETIC_PATH, result)
    return result


def preflight() -> dict[str, Any]:
    _require_clean_tree()
    if not SYNTHETIC_PATH.is_file() or _json(SYNTHETIC_PATH).get("status") != "PASS_SYNTHETIC_END_TO_END_PERSISTENCE_RESUME":
        raise BenchmarkError("Synthetic end-to-end/persistence/resume QA must pass before official preflight")
    resource = _resource_snapshot(initial=True)
    if _sha_file(TRAIN) != TRAIN_SHA:
        raise BenchmarkError("Train hash differs from authorized official input")
    frame = _train_feature_frame()
    reused = _assert_expected_reused_units()
    vectorizer, a02_model = cgeom._c2b_vectorizer()
    vectorizer_audit = cgeom.audit_c2b_vectorizer_compatibility(model=a02_model, frame=frame)
    if vectorizer_audit.get("status") != "PASS_A02_FULLTRAIN_TFIDF_FEATURE_COMPATIBILITY":
        raise BenchmarkError("A02 full-Train vectorizer compatibility check did not pass")
    cache = _make_real_train_cache(frame, vectorizer)
    z_original, p_pos, offset_audit = _verify_position_offset(frame, vectorizer)
    ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    present = cache["rows"]["present_mask"]
    no_known = cache["rows"]["no_known_bigram"]
    _write_npz_once(
        TRAIN_ARRAYS_PATH, record_ids=ids, cycles=cycles,
        present_mask=present, no_known_bigram=no_known,
        z_original=np.asarray(z_original, dtype=np.float64),
        p_pos_original=np.asarray(p_pos, dtype=np.float64),
    )
    events, snapshot, ledger_sha = _ledger_state()
    expected_snapshot = {
        "event_rows": 376, "global_cap": 149,
        "effective_category_caps": {"continuation": 52, "control_replay": 2,
                                     "pilot": 94, "technical_retry": 1},
        "active_reservations_by_category": {"continuation": 50, "control_replay": 2,
                                            "pilot": 91, "technical_retry": 1},
        "active_reservations": 144, "fit_starts": 144,
        "fit_saved": 138, "fit_failed": 6, "fit_pending": 0,
    }
    if snapshot != expected_snapshot or any(e.get("card_id") == CARD_ID for e in events):
        raise BenchmarkError(f"Shared ledger is not at the audited one-fit baseline: {snapshot}")
    if any(e.get("fit_id") == FIT_ID for e in events):
        raise BenchmarkError("The unique REAL E79 Bigram fit ID already exists; refusing duplicate work")
    hashes = _source_hashes()
    data = {
        "schema_version": 1,
        "benchmark_id": CARD_ID,
        "status": "PASS_TARGET_FREE_PREFIT_FROZEN_PENDING_RESERVATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _head(),
        "source_hashes": hashes,
        "input_sha256": {"data/raw/classification/Train.csv": TRAIN_SHA},
        "train_rows": TRAIN_ROWS,
        "train_cycle_range": [47, 79],
        "target_column_loaded": False,
        "target_sha256_computed": False,
        "test_or_sample_opened_by_this_benchmark": False,
        "reused_state": reused,
        "vectorizer_compatibility": vectorizer_audit,
        "fulltrain_original_pos_replay": offset_audit,
        "bigram_real_features": {k: v for k, v in cache.items() if k not in {"matrix", "metadata", "rows"}},
        "cache_paths": {
            "real_matrix": MATRIX_PATH.relative_to(ROOT).as_posix(),
            "train_offset_and_rows": TRAIN_ARRAYS_PATH.relative_to(ROOT).as_posix(),
            "real_matrix_sha256": _sha_file(MATRIX_PATH),
            "train_offset_and_rows_sha256": _sha_file(TRAIN_ARRAYS_PATH),
        },
        "fit_recipe_sha256": _sha_file(REPORT / "RECIPE.json"),
        "current_ledger_sha256": ledger_sha,
        "current_ledger_snapshot": snapshot,
        "resource_snapshot": resource,
        "one_new_gamma_fit_required": True,
        "new_gamma_fit_id": FIT_ID,
    }
    data["freeze_signature"] = _sha_value(data)
    _write_json_once(PREFIT_PATH, data)
    _write_json_once(RUN / "PRE_FIT_FREEZE.json", data)
    del frame, vectorizer, a02_model, cache, z_original, p_pos
    gc.collect()
    return data


def _load_prefit() -> dict[str, Any]:
    data = _json(PREFIT_PATH)
    signature = data.get("freeze_signature")
    body = {key: value for key, value in data.items() if key != "freeze_signature"}
    if signature != _sha_value(body):
        raise BenchmarkError("Pre-fit freeze signature does not verify")
    if data.get("status") != "PASS_TARGET_FREE_PREFIT_FROZEN_PENDING_RESERVATION":
        raise BenchmarkError("Unexpected pre-fit freeze status")
    for relative, expected in data.get("source_hashes", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha_file(path) != expected:
            raise BenchmarkError(f"Frozen source changed after preflight: {relative}")
    if _sha_file(MATRIX_PATH) != data["cache_paths"]["real_matrix_sha256"]:
        raise BenchmarkError("Frozen REAL bigram matrix file hash changed")
    if _sha_file(TRAIN_ARRAYS_PATH) != data["cache_paths"]["train_offset_and_rows_sha256"]:
        raise BenchmarkError("Frozen Train offset/row bundle hash changed")
    return data


def reserve() -> dict[str, Any]:
    _require_clean_tree()
    freeze = _load_prefit()
    if not _git("ls-files", "--error-unmatch", str(PREFIT_PATH.relative_to(ROOT))):
        raise BenchmarkError("Pre-fit freeze must be committed before budget reservation")
    events, snapshot, ledger_sha = _ledger_state()
    if ledger_sha != freeze.get("current_ledger_sha256") or snapshot != freeze.get("current_ledger_snapshot"):
        raise BenchmarkError("Shared ledger changed after the pre-fit freeze; stop and audit before reserving")
    if any(e.get("card_id") == CARD_ID or e.get("reservation_key") == RESERVATION_KEY for e in events):
        raise BenchmarkError("Unique reservation namespace already exists; refusing duplicate reservation")
    if any(e.get("fit_id") == FIT_ID for e in events):
        raise BenchmarkError("Unique fit ID already appears in the shared ledger")
    previous_caps = snapshot["effective_category_caps"]
    new_caps = dict(previous_caps)
    new_caps["pilot"] += 1
    new_cap = int(snapshot["global_cap"]) + 1
    if sum(new_caps.values()) != new_cap:
        raise BenchmarkError("One-slot cap amendment does not balance across categories")
    if (snapshot["fit_pending"] != 0
            or snapshot["active_reservations"] + 1 > new_cap
            or snapshot["active_reservations_by_category"]["pilot"] + 1 > new_caps["pilot"]):
        raise BenchmarkError("Actual shared budget cannot safely reserve exactly one new fit")
    amendment = {
        "event": "BUDGET_CAP_AMENDMENT", "amendment_id": AMENDMENT_ID,
        "card_id": CARD_ID, "stage": STAGE,
        "previous_global_cap": int(snapshot["global_cap"]), "global_cap": new_cap,
        "previous_caps": previous_caps, "effective_caps": new_caps,
        "added_fits": 1, "source_commit": freeze["source_commit"],
        "freeze_signature": freeze["freeze_signature"],
        "authorization": "reports/optimization_v3/bigram_position_real_e79_v1/BENCHMARK_AUTHORIZATION.md",
        "reason": "One user-authorized REAL bigram-position E79 gamma fit; no released slot is reused.",
    }
    offset._append_ledger(amendment, path=LEDGER, official=True)
    reservation = {
        "event": "CARD_FITS_RESERVED", "reservation_key": RESERVATION_KEY,
        "card_id": CARD_ID, "stage": STAGE, "category": "pilot", "fits": 1,
        "fit_ids": [FIT_ID], "amendment_id": AMENDMENT_ID,
        "source_commit": freeze["source_commit"], "freeze_signature": freeze["freeze_signature"],
    }
    offset._append_ledger(reservation, path=LEDGER, official=True)
    new_events, new_snapshot, new_sha = _ledger_state()
    expected = dict(snapshot)
    expected.update({
        "event_rows": snapshot["event_rows"] + 2,
        "global_cap": new_cap,
        "effective_category_caps": new_caps,
        "active_reservations_by_category": {
            **snapshot["active_reservations_by_category"],
            "pilot": snapshot["active_reservations_by_category"]["pilot"] + 1,
        },
        "active_reservations": snapshot["active_reservations"] + 1,
    })
    if new_snapshot != expected:
        raise BenchmarkError(f"Post-reservation ledger reconciliation failed: {new_snapshot}")
    result = {
        "status": "RESERVED_ONE_FIT",
        "amendment": amendment,
        "reservation": reservation,
        "prior_ledger_sha256": ledger_sha,
        "final_ledger_sha256": new_sha,
        "ledger_snapshot": new_snapshot,
        "new_fit_id": FIT_ID,
        "no_fit_started": True,
    }
    _write_json_once(RESERVATION_PATH, result)
    _write_json_once(RUN / "RESERVATION.json", result)
    return result


def _validate_reservation(freeze: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    events, snapshot, ledger_sha = _ledger_state()
    matches = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
               and e.get("reservation_key") == RESERVATION_KEY]
    if (len(matches) != 1 or matches[0].get("fit_ids") != [FIT_ID]
            or matches[0].get("freeze_signature") != freeze.get("freeze_signature")):
        raise BenchmarkError("Unique one-fit reservation does not match frozen recipe")
    amendment = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                 and e.get("amendment_id") == AMENDMENT_ID]
    if len(amendment) != 1 or amendment[0].get("added_fits") != 1:
        raise BenchmarkError("Unique one-fit cap amendment is missing or malformed")
    own = [e for e in events if e.get("fit_id") == FIT_ID]
    if any(e.get("reservation_key") != RESERVATION_KEY for e in own):
        raise BenchmarkError("Fit ID appears under a conflicting reservation")
    return events, snapshot, ledger_sha


def _load_fit_bundle() -> tuple[np.ndarray, dict[str, Any]]:
    if not GAMMA_PATH.is_file() or not RUN_RECEIPT_PATH.is_file():
        raise BenchmarkError("Durable fitted gamma or receipt is absent")
    receipt = _json(RUN_RECEIPT_PATH)
    if (receipt.get("fit_id") != FIT_ID
            or receipt.get("matrix_file_sha256") != _sha_file(MATRIX_PATH)
            or receipt.get("artifact_sha256") != _sha_file(GAMMA_PATH)):
        raise BenchmarkError("Saved fit artifact receipt does not bind to the frozen input")
    with np.load(GAMMA_PATH, allow_pickle=False) as archive:
        gamma = np.asarray(archive["gamma"], dtype=np.float64).reshape(-1).copy()
    if offset.sha256_array(gamma) != receipt.get("gamma_sha256") or not np.isfinite(gamma).all():
        raise BenchmarkError("Saved gamma logical hash or finite-value check failed")
    return gamma, receipt


def _record_fit_failure(error: Exception, start_event: Mapping[str, Any]) -> None:
    events, _, _ = _ledger_state()
    terminal = [e for e in events if e.get("fit_id") == FIT_ID
                and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if not terminal:
        offset._append_ledger({
            "event": "FIT_FAILED", "card_id": CARD_ID, "stage": STAGE,
            "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID, "cutoff": 79,
            "component": "REAL_BIGRAM_POSITION_LBFGS_GAMMA",
            "error": f"{type(error).__name__}: {error}",
            "amendment_id": AMENDMENT_ID,
            "source_commit": start_event.get("source_commit"),
        }, path=LEDGER, official=True)


def fit() -> dict[str, Any]:
    _require_clean_tree()
    freeze = _load_prefit()
    if not _git("ls-files", "--error-unmatch", str(RESERVATION_PATH.relative_to(ROOT))):
        raise BenchmarkError("One-fit reservation must be committed before training")
    events, snapshot, ledger_sha = _validate_reservation(freeze)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == FIT_ID]
    if len(starts) > 1 or len(terminals) > 1:
        raise BenchmarkError("Duplicate fit lifecycle events exist; refusing any new work")
    if terminals:
        if terminals[0]["event"] == "FIT_FAILED":
            raise BenchmarkError("The unique authorized fit already failed; no retry is permitted")
        gamma, receipt = _load_fit_bundle()
        return {"status": "FIT_SAVED_RESUMED_NO_REFIT", "fit_id": FIT_ID,
                "gamma_sha256": offset.sha256_array(gamma), "receipt_sha256": _sha_file(RUN_RECEIPT_PATH),
                "ledger_sha256": ledger_sha, "fit_pending": snapshot["fit_pending"],
                "receipt": receipt}
    if starts:
        # Interrupted output is recoverable only when the durable gamma+receipt exist.
        if GAMMA_PATH.is_file() and RUN_RECEIPT_PATH.is_file():
            gamma, receipt = _load_fit_bundle()
            offset._append_ledger({
                "event": "FIT_SAVED", "card_id": CARD_ID, "stage": STAGE,
                "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID, "cutoff": 79,
                "component": "REAL_BIGRAM_POSITION_LBFGS_GAMMA",
                "receipt_sha256": _sha_file(RUN_RECEIPT_PATH), "amendment_id": AMENDMENT_ID,
                "recovered_durable_output": True,
            }, path=LEDGER, official=True)
            return {"status": "FIT_SAVED_RECOVERED_NO_REFIT", "fit_id": FIT_ID,
                    "gamma_sha256": offset.sha256_array(gamma), "receipt_sha256": _sha_file(RUN_RECEIPT_PATH),
                    "receipt": receipt}
        _record_fit_failure(BenchmarkError("Fit start exists without durable output; retry prohibited"), starts[0])
        raise BenchmarkError("Interrupted fit has no durable result; slot consumed and retry prohibited")
    if snapshot["fit_pending"] != 0:
        raise BenchmarkError(f"Shared ledger has another pending fit: {snapshot}")
    resource = _resource_snapshot(initial=False)
    if resource["available_ram_bytes"] < INITIAL_FREE_MIN:
        raise BenchmarkError("Available RAM fell below the project 4 GiB pre-fit minimum")
    if _sha_file(MATRIX_PATH) != freeze["cache_paths"]["real_matrix_sha256"]:
        raise BenchmarkError("REAL bigram training matrix hash differs from pre-fit freeze")
    if _sha_file(TRAIN_ARRAYS_PATH) != freeze["cache_paths"]["train_offset_and_rows_sha256"]:
        raise BenchmarkError("Train offset/row bundle hash differs from pre-fit freeze")
    with np.load(TRAIN_ARRAYS_PATH, allow_pickle=False) as archive:
        rows = {key: archive[key].copy() for key in archive.files}
    matrix = sparse.load_npz(MATRIX_PATH).tocsr()
    if (len(rows["record_ids"]) != TRAIN_ROWS or matrix.shape[0] != TRAIN_ROWS
            or offset.sha256_csr(matrix) != freeze["bigram_real_features"]["matrix_logical_sha256"]
            or rows["z_original"].shape != (TRAIN_ROWS,)):
        raise BenchmarkError("Saved full-Train fit inputs fail row or matrix integrity checks")
    if GAMMA_PATH.exists() or RUN_RECEIPT_PATH.exists():
        raise BenchmarkError("Unexpected prior fit artifact exists without ledger terminal event")
    start_event = {
        "event": "FIT_STARTED", "card_id": CARD_ID, "stage": STAGE,
        "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID, "cutoff": 79,
        "component": "REAL_BIGRAM_POSITION_LBFGS_GAMMA", "fit_kind": "FULLTRAIN_PREDICTOR",
        "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
        "source_commit": _head(), "freeze_signature": freeze["freeze_signature"],
        "fit_population": {
            "rows": TRAIN_ROWS, "cycles": [47, 79],
            "record_ids_sha256": offset.sha256_array(rows["record_ids"]),
            "cycles_sha256": offset.sha256_array(rows["cycles"]),
            "target_source": "official Train artifact_signal, read only after FIT_STARTED",
            "feature_matrix_sha256": offset.sha256_csr(matrix),
            "offset_sha256": offset.sha256_array(rows["z_original"]),
        },
        "resource_snapshot": resource,
    }
    offset._append_ledger(start_event, path=LEDGER, official=True)
    start_time = datetime.now(timezone.utc).isoformat()
    try:
        # First official target read occurs after the durable FIT_STARTED event.
        target_frame = pd.read_csv(TRAIN, usecols=["artifact_signal"])
        y = pd.to_numeric(target_frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if len(y) != TRAIN_ROWS or not np.isin(y, (0, 1)).all() or set(np.unique(y)) != {0, 1}:
            raise BenchmarkError("Official full-Train target is not exactly 50,608 binary rows")
        with threadpool_limits(limits=THREADS):
            gamma, optimizer = fit_offset_coefficients(
                matrix, rows["z_original"], y, c=1.0
            )
        if optimizer.get("success") is not True or not np.isfinite(gamma).all():
            raise BenchmarkError(f"Single authorized gamma fit failed acceptance: {optimizer}")
        npz_arrays = {"gamma": np.asarray(gamma, dtype=np.float64)}
        _write_npz_once(GAMMA_PATH, **npz_arrays)
        receipt = {
            "schema_version": 1,
            "benchmark_id": CARD_ID,
            "fit_id": FIT_ID,
            "status": "FIT_SAVED",
            "fit_count": 1,
            "retry_count": 0,
            "source_commit": start_event["source_commit"],
            "pre_fit_freeze_sha256": _sha_file(PREFIT_PATH),
            "reservation_sha256": _sha_file(RESERVATION_PATH),
            "train_path": TRAIN.relative_to(ROOT).as_posix(),
            "train_sha256": TRAIN_SHA,
            "fit_rows": TRAIN_ROWS,
            "fit_cycles": [47, 79],
            "record_ids_sha256": offset.sha256_array(rows["record_ids"]),
            "target_sha256": offset.sha256_array(y),
            "matrix_path": MATRIX_PATH.relative_to(ROOT).as_posix(),
            "matrix_file_sha256": _sha_file(MATRIX_PATH),
            "matrix_logical_sha256": offset.sha256_csr(matrix),
            "matrix_shape": list(matrix.shape),
            "offset_sha256": offset.sha256_array(rows["z_original"]),
            "gamma_path": GAMMA_PATH.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha_file(GAMMA_PATH),
            "gamma_sha256": offset.sha256_array(gamma),
            "target_read_after_fit_started": True,
            "test_features_used_for_fit": False,
            "test_labels_loaded": False,
            "sample_submission_opened": False,
            "optimizer": optimizer,
            "resource_snapshot": resource,
            "started_at_utc": start_time,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["receipt_signature"] = _sha_value(receipt)
        _write_json_once(RUN_RECEIPT_PATH, receipt)
        _write_json_once(FIT_RECEIPT_PATH, receipt)
        receipt_sha = _sha_file(RUN_RECEIPT_PATH)
        offset._append_ledger({
            "event": "FIT_SAVED", "card_id": CARD_ID, "stage": STAGE,
            "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID, "cutoff": 79,
            "component": "REAL_BIGRAM_POSITION_LBFGS_GAMMA",
            "receipt_sha256": receipt_sha, "amendment_id": AMENDMENT_ID,
            "source_commit": start_event["source_commit"],
        }, path=LEDGER, official=True)
        return {"status": "FIT_SAVED", "fit_id": FIT_ID, "gamma_sha256": receipt["gamma_sha256"],
                "receipt_sha256": receipt_sha, "optimizer": optimizer,
                "target_sha256": receipt["target_sha256"]}
    except Exception as exc:
        _record_fit_failure(exc, start_event)
        raise


def freeze_trained() -> dict[str, Any]:
    freeze = _load_prefit()
    events, snapshot, ledger_sha = _validate_reservation(freeze)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    saves = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == FIT_ID]
    fails = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == FIT_ID]
    if len(starts) != 1 or len(saves) != 1 or fails or snapshot["fit_pending"] != 0:
        raise BenchmarkError(f"Cannot freeze fitted state: {snapshot}")
    gamma, receipt = _load_fit_bundle()
    result = {
        "schema_version": 1,
        "benchmark_id": CARD_ID,
        "status": "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE",
        "fit_id": FIT_ID,
        "fit_count": 1,
        "historical_pilot_verdict_changed": False,
        "prepared_source_commit": freeze["source_commit"],
        "fit_source_commit": starts[0].get("source_commit"),
        "pre_fit_freeze_sha256": _sha_file(PREFIT_PATH),
        "fit_receipt_sha256": _sha_file(RUN_RECEIPT_PATH),
        "gamma_file_sha256": _sha_file(GAMMA_PATH),
        "gamma_sha256": offset.sha256_array(gamma),
        "train_sha256": TRAIN_SHA,
        "train_rows": TRAIN_ROWS,
        "fit_target_sha256": receipt["target_sha256"],
        "fit_source_hashes": freeze["source_hashes"],
        "test_features_opened_by_this_benchmark": False,
        "test_labels_loaded": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "ledger_sha256": ledger_sha,
        "ledger_snapshot": snapshot,
    }
    result["freeze_signature"] = _sha_value(result)
    _write_json_once(TRAINED_FREEZE_PATH, result)
    _write_json_once(RUN / "TRAINED_STATE_FREEZE.json", result)
    return result


def _load_trained_freeze() -> dict[str, Any]:
    data = _json(TRAINED_FREEZE_PATH)
    signature = data.get("freeze_signature")
    body = {key: value for key, value in data.items() if key != "freeze_signature"}
    if (signature != _sha_value(body)
            or data.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or data.get("fit_id") != FIT_ID or data.get("test_labels_loaded") is not False):
        raise BenchmarkError("Trained-state freeze does not verify or is not closed before Test")
    head_blob = subprocess.run(
        ["git", "show", f"HEAD:{TRAINED_FREEZE_PATH.relative_to(ROOT).as_posix()}"],
        cwd=ROOT, capture_output=True, check=False,
    )
    if head_blob.returncode or hashlib.sha256(head_blob.stdout).hexdigest() != _sha_file(TRAINED_FREEZE_PATH):
        raise BenchmarkError("Trained-state freeze is not committed at the current HEAD")
    return data


def infer() -> dict[str, Any]:
    _require_clean_tree()
    trained_freeze = _load_trained_freeze()
    freeze = _load_prefit()
    if _source_hashes() != freeze["source_hashes"]:
        raise BenchmarkError("Training/inference code or recipe changed after the pre-fit freeze")
    gamma, fit_receipt = _load_fit_bundle()
    if trained_freeze.get("gamma_sha256") != offset.sha256_array(gamma):
        raise BenchmarkError("Inference gamma differs from committed trained-state freeze")
    # The Test file is not opened until the committed trained-state freeze above passes.
    if _sha_file(TEST) != TEST_SHA:
        raise BenchmarkError("Official Test file hash differs from the frozen source")
    usecols = ["record_id", "cycle_code", *cgeom.EXPANDED_FEATURES, "signal_stream"]
    test = pd.read_csv(TEST, usecols=usecols, dtype={"record_id": str})
    _validate_identity_features(test, train=False)
    ids = test["record_id"].astype(str).to_numpy(dtype=str)
    cycles = test["cycle_code"].to_numpy(dtype=np.int64)
    vectorizer, _a02_model = cgeom._c2b_vectorizer()
    test_base, test_unigram_position, test_audit = cgeom._c2b_matrices(
        test, vectorizer=vectorizer, train_side=True
    )
    if test_unigram_position is None:
        raise BenchmarkError("Test unigram-position feature transform was not produced")
    bigram_blocks, test_metadata = build_bigram_blocks(
        cgeom.positional._text_values(test["signal_stream"])[0], vectorizer
    )
    test_matrix = sparse.csr_matrix(bigram_blocks["real"], dtype=np.float64)
    c2b_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[3]
    pos_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[6]
    c1_state_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[4]
    c1_model_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[5]
    bag = cgeom._load_unit(c2b_id)["estimator"]
    original_gamma = np.asarray(cgeom._load_unit(pos_id)["gamma"], dtype=np.float64).reshape(-1)
    _z_bag, z_original = reconstruct_original_pos_logit(
        test_base, bag.coef_[0], bag.intercept_, test_unigram_position, original_gamma
    )
    p_pos = np.asarray(expit(z_original), dtype=np.float64)

    batch = _json(BATCH_MANIFEST)
    inference_receipt = _json(TEST_INFERENCE_RECEIPT)
    if (batch.get("prediction_bundle_sha256") != _sha_file(TEST_BUNDLE)
            or inference_receipt.get("test_target_loaded") is not False
            or inference_receipt.get("kaggle_contacted") is not False):
        raise BenchmarkError("Frozen full-Train Test probability bundle receipt failed")
    with np.load(TEST_BUNDLE, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {"record_ids", "missing_mask", "a02_probability", "c1b_missing_probability",
                "c2b_probability", "posoffset_direct_probability", "posoffset_probability"}
    if not required.issubset(arrays):
        raise BenchmarkError(f"Frozen Test probability bundle is missing required arrays: {required - set(arrays)}")
    if not np.array_equal(ids, arrays["record_ids"].astype(str)):
        raise BenchmarkError("Official Test IDs/order differ from frozen A02-CGEOM probability bundle")
    missing = np.asarray(test_audit["missing"], dtype=bool)
    frozen_missing = np.asarray(arrays["missing_mask"], dtype=bool)
    if not np.array_equal(missing, frozen_missing):
        raise BenchmarkError("Test stream availability differs from frozen full-Train route")
    p_a02 = np.asarray(arrays["a02_probability"], dtype=np.float64)
    p_p = np.asarray(arrays["posoffset_probability"], dtype=np.float64)
    c1_estimator = cgeom._load_unit(c1_model_id)["estimator"]
    c1_state = cgeom._load_unit(c1_state_id)
    c1_test = cgeom.positional.transform_c1b_structured(c1_state, cgeom._features(test))
    p_missing = cgeom.positional.positive_probability(c1_estimator, c1_test)
    if not np.array_equal(p_missing, arrays["c1b_missing_probability"]):
        raise BenchmarkError("Replayed full-Train C1_B HGB probabilities differ from frozen Test bundle")
    if not np.array_equal(p_p[missing], p_missing[missing]):
        raise BenchmarkError("Frozen POS route does not preserve exact C1_B missing fallback")
    ppos_bundle = np.asarray(arrays["posoffset_direct_probability"], dtype=np.float64)
    ppos_diff = float(np.max(np.abs(p_pos[~missing] - ppos_bundle[~missing]), initial=0.0))
    if ppos_diff > TOL:
        raise BenchmarkError(f"Reconstructed original unigram POS differs from frozen route by {ppos_diff}")
    if any(len(v) != TEST_ROWS for v in (p_a02, p_p, p_missing, z_original)):
        raise BenchmarkError("Inference probabilities or offsets do not align to all Test rows")
    if (not np.isfinite(p_a02).all() or np.any((p_a02 < 0) | (p_a02 > 1))
            or not np.isfinite(p_p).all() or np.any((p_p < 0) | (p_p > 1))):
        raise BenchmarkError("Frozen Test probabilities contain non-finite/out-of-range values")

    mix50 = 0.5 * p_pos + 0.5 * p_a02
    mix50[missing] = p_p[missing]
    present = ~missing
    no_known = np.asarray(test_metadata["known_occurrence_count"] == 0, dtype=bool)
    zero = zero_gamma_equivalence(
        test_matrix, z_original, present_mask=present, p_a02=p_a02,
        p_p=p_p, mix50_original=mix50,
    )
    if not zero["pass"]:
        raise BenchmarkError(f"E79 Test zero-gamma route does not reproduce original MIX50: {zero}")
    _q, probability = route_candidate(
        test_matrix, z_original, gamma, present_mask=present,
        no_known_bigram=no_known, p_pos_original=p_pos,
        p_a02=p_a02, p_p=p_p, mix50_original=mix50,
    )
    labels = (probability >= THRESHOLD).astype(np.int8)
    output = pd.DataFrame({"record_id": ids, "artifact_signal": labels})
    expected_columns = ["record_id", "artifact_signal"]
    if (output.columns.tolist() != expected_columns or len(output) != TEST_ROWS
            or output["record_id"].duplicated().any()
            or output["artifact_signal"].dtype.kind not in "iu"
            or not output["artifact_signal"].isin([0, 1]).all()):
        raise BenchmarkError("Submission candidate fails row/order/schema/binary checks")
    if CSV.exists():
        raise BenchmarkError(f"Refusing to overwrite an existing candidate CSV: {CSV}")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = CSV.with_name(CSV.name + f".{os.getpid()}.partial")
    try:
        output.to_csv(temp_csv, index=False)
        with temp_csv.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.link(temp_csv, CSV)
    finally:
        temp_csv.unlink(missing_ok=True)
    csv_hash = _sha_file(CSV)
    result = {
        "schema_version": 1,
        "benchmark_id": CARD_ID,
        "status": "PASS_TEST_INFERENCE_QA_NOT_SUBMITTED",
        "test_path": TEST.relative_to(ROOT).as_posix(),
        "test_sha256": TEST_SHA,
        "test_rows": TEST_ROWS,
        "test_id_order_sha256": offset.sha256_array(ids),
        "cycle_range": [int(cycles.min()), int(cycles.max())],
        "all_cycles_80_91_present": True,
        "missing_rows": int(missing.sum()),
        "present_rows": int(present.sum()),
        "bigram_vocabulary_size": int(test_metadata["bigram_vocabulary_size"]),
        "known_bigram_rows": int(np.count_nonzero(test_metadata["known_occurrence_count"])),
        "no_known_bigram_present_rows": int(np.count_nonzero(present & no_known)),
        "original_pos_replay_max_abs_error_present": ppos_diff,
        "zero_gamma_equivalence": zero,
        "missing_c1b_fallback_bit_exact": bool(np.array_equal(probability[missing], p_p[missing])),
        "no_known_bigram_mix50_fallback_bit_exact": bool(np.array_equal(
            probability[present & no_known], mix50[present & no_known])),
        "csv_path": CSV.relative_to(ROOT).as_posix(),
        "csv_sha256": csv_hash,
        "csv_rows": len(output),
        "csv_columns": expected_columns,
        "csv_labels_integer_0_1": True,
        "predicted_positive_count": int(labels.sum()),
        "threshold": THRESHOLD,
        "sample_submission_opened": False,
        "test_labels_loaded": False,
        "kaggle_contacted": False,
        "public_score": None,
        "fit_receipt_sha256": _sha_file(RUN_RECEIPT_PATH),
        "trained_state_freeze_sha256": _sha_file(TRAINED_FREEZE_PATH),
        "source_commit": _head(),
    }
    _write_json_once(INFERENCE_QA_PATH, result)
    _write_json_once(RUN / "INFERENCE_QA.json", result)
    finalize(result)
    return result


def finalize(inference: Mapping[str, Any]) -> dict[str, Any]:
    freeze = _load_prefit()
    gamma, receipt = _load_fit_bundle()
    events, snapshot, ledger_sha = _ledger_state()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    saves = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") == FIT_ID]
    fails = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") == FIT_ID]
    if len(starts) != 1 or len(saves) != 1 or fails or snapshot["fit_pending"]:
        raise BenchmarkError(f"Final fit accounting reconciliation failed: {snapshot}")
    accounting = {
        "status": "PASS_ONE_NEW_FIT_APPEND_ONLY_ACCOUNTING",
        "benchmark_id": CARD_ID,
        "fit_id": FIT_ID,
        "authorized_fits": 1,
        "fit_starts": len(starts),
        "fit_saved": len(saves),
        "fit_failed": len(fails),
        "fit_pending": snapshot["fit_pending"],
        "retry_count": 0,
        "ledger_path": LEDGER.relative_to(ROOT).as_posix(),
        "ledger_sha256": ledger_sha,
        "ledger_snapshot": snapshot,
        "fit_receipt_sha256": _sha_file(RUN_RECEIPT_PATH),
        "gamma_sha256": offset.sha256_array(gamma),
        "no_released_slot_reused": True,
    }
    _write_json_once(ACCOUNTING_PATH, accounting)
    manifest = {
        "schema_version": 1,
        "benchmark_id": CARD_ID,
        "status": "EXPLORATORY_BENCHMARK_PREPARED_NOT_SUBMITTED",
        "authorization_path": "reports/optimization_v3/bigram_position_real_e79_v1/BENCHMARK_AUTHORIZATION.md",
        "recipe_path": "reports/optimization_v3/bigram_position_real_e79_v1/RECIPE.json",
        "pre_fit_freeze_sha256": _sha_file(PREFIT_PATH),
        "trained_state_freeze_sha256": _sha_file(TRAINED_FREEZE_PATH),
        "fit_receipt_sha256": _sha_file(RUN_RECEIPT_PATH),
        "budget_accounting_sha256": _sha_file(ACCOUNTING_PATH),
        "input_hashes": {"classification_train": TRAIN_SHA, "classification_test": TEST_SHA},
        "fit_count": 1,
        "retry_count": 0,
        "gamma_sha256": offset.sha256_array(gamma),
        "csv_path": inference["csv_path"],
        "csv_sha256": inference["csv_sha256"],
        "test_labels_loaded": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
        "public_score": None,
        "historical_bigram_pilot_gate_verdict_changed": False,
        "historical_pos_moments_benchmark_touched": False,
        "source_hashes": freeze["source_hashes"],
        "source_commit": inference["source_commit"],
    }
    _write_json_once(MANIFEST_PATH, manifest)
    _write_json_once(RUN / "BENCHMARK_MANIFEST.json", manifest)
    lines = [
        "# CLS Bigram Position REAL E79 — exploratory benchmark",
        "",
        "**Status: candidate prepared for Sol review; no Kaggle upload.**",
        "",
        "The benchmark reused the verified full-Train C2_B, original unigram POS,",
        "A02, and C1_B states, then ran exactly one new REAL bigram-position gamma",
        "fit on all 50,608 official Train rows. No BAG/SHUFFLED E79 fit, retry,",
        "test-label read, SampleSubmission access, or Kaggle call occurred.",
        "",
        f"- Candidate CSV: `{inference['csv_path']}`",
        f"- CSV SHA-256: `{inference['csv_sha256']}`",
        f"- Full-Train fit ID: `{FIT_ID}`",
        f"- Gamma SHA-256: `{offset.sha256_array(gamma)}`",
        f"- Test rows/positive predictions: {inference['csv_rows']} / {inference['predicted_positive_count']}",
        f"- Missing-stream rows: {inference['missing_rows']}; present rows with no known bigram: {inference['no_known_bigram_present_rows']}",
        f"- Zero-gamma equivalence: {'PASS' if inference['zero_gamma_equivalence']['pass'] else 'FAIL'}; maximum present difference {inference['zero_gamma_equivalence']['max_abs_present_difference']:.3g}",
        f"- Ledger: 1 start, 1 saved, 0 failed, 0 pending; SHA-256 `{ledger_sha}`",
        "- Test labels were not read; the candidate has no public score and remains unsubmitted.",
        "",
        "## Interpretation boundary",
        "",
        "This is an exploratory E79 inference artifact, not a new internal-validation",
        "result and not a qualified champion. It does not alter the failed diagnostic",
        "verdict of the E67/E75 Bigram Position pilot or the separate Moments lane.",
        "",
        "## One zero-fit explanation for a later review",
        "",
        "In the existing pilot, E75 W6 REAL was 0.001482 Macro F1 below MIX50 and",
        "also below both BAG and SHUFFLED. Before authorizing another fit, inspect",
        "whether the saved E75 REAL gamma's prediction changes are concentrated",
        "near the 0.50 threshold and whether those changes increase false positives",
        "or false negatives. This is a prediction/error decomposition only; it does",
        "not justify a new fit or change the historical gate.",
    ]
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"accounting": accounting, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthetic", "preflight", "reserve", "fit", "freeze-trained", "infer"))
    args = parser.parse_args(argv)
    functions = {
        "synthetic": synthetic,
        "preflight": preflight,
        "reserve": reserve,
        "fit": fit,
        "freeze-trained": freeze_trained,
        "infer": infer,
    }
    result = functions[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
