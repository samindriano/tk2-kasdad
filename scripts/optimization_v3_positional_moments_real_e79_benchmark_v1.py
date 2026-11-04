"""One-shot, frozen POS_MOMENTS E79 exploratory benchmark.

This namespace is deliberately separate from POS_MOMENTS_V1.  It reuses the
verified full-Train C2_B and A02 artifacts, fits one gamma on official Train,
then creates one local Test-inference CSV for Sol review.  It never opens
SampleSubmission or Test labels and never contacts Kaggle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_a02_cgeom_benchmark_v1 as cgeom
from scripts import optimization_v3_positional_moments_v1_prelabel_v2 as scalar_replay
from src.models.positional_bag_offset_v1 import fit_offset_coefficients
from src.models.positional_moments_v1 import build_moment_matrix, sha256_array, sha256_csr, sha256_file


BENCHMARK_ID = "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK"
FIT_ID = "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK_GAMMA"
RESERVATION_KEY = f"{BENCHMARK_ID}:REAL_GAMMA_FULLTRAIN"
AMENDMENT_ID = "POS_MOMENTS_REAL_E79_EXPLORATORY_BENCHMARK_CAP143_2026_11_04"
TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
TEST_SHA = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
A02_FREEZE_SHA = "ab0307e6e1c9778ed0f091a3d6ed55e0ac8705ae31b9cb29b9f083e46b5c2b82"
A02_SOURCE_TEST_NPZ_SHA = "c267a4b998aca64982b6f63b10bedb2f3b3061007166db7731d221788ac89e47"
PREDICTION_BUNDLE_SHA = "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"
BASE_MATRIX_SHA = "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b"
EXPECTED_TARGET_SHA = "180f7f9dc5462c4be7d683f1400349113ae9025a13249105423feaa45e827dc4"
BASE_LEDGER_SHA = "6daf83a71de6b42f73e0ba25d9532a06250cd9738f0a23a55e78bd1b93d163bd"
BASE_LEDGER_ROWS = 357
OLD_GLOBAL_CAP = 142
NEW_GLOBAL_CAP = 143
OLD_CAPS = {"continuation": 51, "control_replay": 2, "pilot": 88, "technical_retry": 1}
NEW_CAPS = {"continuation": 52, "control_replay": 2, "pilot": 88, "technical_retry": 1}
RELEASE_KEY = "POS_CONTRAST_V1:CONDITIONAL_E79_AFTER_SOL_REVIEW"
RELEASE_EVENT_SHA = "c474982940a97bd55a31da6276aebd3daa3039699e75674499ec86e9fdafb521"
MAX_RSS = 6 * 1024**3
MIN_INITIAL_FREE = 4 * 1024**3
MIN_RUNNING_FREE = 2 * 1024**3
THREADS = 2

REPORT = ROOT / "reports/optimization_v3/positional_moments_real_e79_benchmark_v1"
RUN = ROOT / "experiments/runs/optimization_v3_positional_moments_real_e79_benchmark_v1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
TEST = ROOT / "data/raw/classification/Test.csv"
LEDGER = cgeom.LEDGER
A02_FREEZE = ROOT / "reports/optimization_v3/a02_cgeom_benchmark_v1/E79_TRAINED_FREEZE.json"
A02_SOURCE_TEST_NPZ = cgeom.A02_TEST_NPZ
PREDICTION_BUNDLE = cgeom.OUTPUT_NPZ
CSV = ROOT / "submissions/optimization_v3/positional_moments_real_e79_benchmark_v1/classification_pos_moments_real_e79_exploratory.csv"
PREFIT_AUDIT = REPORT / "PREFIT_AUDIT_R2.json"
SYNTHETIC_QA = REPORT / "SYNTHETIC_QA_R2.json"
PREFIT_FREEZE = REPORT / "PRE_FIT_FREEZE.json"
RESERVATION_REPORT = REPORT / "RESERVATION.json"
FIT_RECEIPT = REPORT / "FIT_RECEIPT.json"
TRAINED_FREEZE = REPORT / "TRAINED_STATE_FREEZE.json"
QA_REPORT = REPORT / "INFERENCE_QA.json"
MANIFEST = REPORT / "BENCHMARK_MANIFEST.json"
INFERENCE_BUILD = REPORT / "INFERENCE_BUILD.json"
W_CACHE = RUN / "prefit_r2" / "W_real.npz"
ARRAY_CACHE = RUN / "prefit_r2" / "features.npz"
GAMMA_PATH = RUN / "fit" / "gamma.npz"

SOURCE_PATHS = (
    "scripts/optimization_v3_positional_moments_real_e79_benchmark_v1.py",
    "scripts/verify_optimization_v3_positional_moments_real_e79_benchmark_v1.py",
    "scripts/optimization_v3_positional_moments_v1_prelabel_v2.py",
    "src/models/positional_moments_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/positional_text_features.py",
    "src/models/classification_models.py",
    "src/models/classification_a02_neighbors.py",
    "scripts/optimization_v3_a02_cgeom_benchmark_v1.py",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/E79_TRAINED_FREEZE.json",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/BATCH_MANIFEST.json",
    "reports/optimization_v3/a02_continuation_v1/submission_manifest.json",
    "reports/optimization_v3/positional_moments_v1/RECIPE.json",
    "reports/optimization_v3/positional_moments_v1/FREEZE.json",
    "reports/optimization_v3/positional_moments_v1/SAFE_EXPORT_PRELABEL_VERIFICATION_V2.json",
    "reports/optimization_v3/positional_moments_v1/SYNTHETIC_PRELABEL_REPLAY_V2.json",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/RECIPE.json",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/BENCHMARK_AUTHORIZATION.md",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/PREFIT_AUDIT.json",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/PREFIT_AUDIT_CORRECTION.md",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/SYNTHETIC_QA.json",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/SYNTHETIC_QA_R2.json",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/PREFIT_AUDIT_R2.json",
    "tests/test_optimization_v3_positional_moments_real_e79_benchmark_v1.py",
)


class BenchmarkError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise BenchmarkError(f"Expected a JSON object: {path}")
    return result


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise BenchmarkError(f"Refusing to replace immutable output: {path}")
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise BenchmarkError(f"Output appeared concurrently: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _save_npz_once(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BenchmarkError(f"Refusing to overwrite NPZ state: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise BenchmarkError(f"NPZ appeared concurrently: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _require_clean_tree() -> None:
    if _git("status", "--porcelain"):
        raise BenchmarkError("A clean worktree is required at this protocol boundary")


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in SOURCE_PATHS:
        path = ROOT / rel
        if not path.is_file():
            raise BenchmarkError(f"Required frozen source is absent: {rel}")
        result[rel] = sha256_file(path)
    return result


def _assert_sources(expected: Mapping[str, str]) -> None:
    if _source_hashes() != dict(expected):
        raise BenchmarkError("Frozen source hashes changed")


def _resource() -> dict[str, int]:
    process = psutil.Process()
    return {"process_rss_bytes": int(process.memory_info().rss),
            "available_ram_bytes": int(psutil.virtual_memory().available)}


def _assert_resource(*, initial: bool, estimated_incremental: int = 0) -> dict[str, int]:
    state = _resource()
    min_available = MIN_INITIAL_FREE if initial else MIN_RUNNING_FREE
    projected_rss = state["process_rss_bytes"] + estimated_incremental
    projected_available = state["available_ram_bytes"] - estimated_incremental
    if (projected_rss > MAX_RSS or state["available_ram_bytes"] < min_available
            or projected_available < MIN_RUNNING_FREE):
        raise BenchmarkError(f"Resource gate failed: {state}, estimate={estimated_incremental}")
    return {**state, "estimated_incremental_memory_bytes": int(estimated_incremental),
            "projected_rss_bytes": int(projected_rss),
            "projected_available_ram_bytes": int(projected_available),
            "rss_ceiling_bytes": MAX_RSS, "initial_free_min_bytes": MIN_INITIAL_FREE,
            "running_free_reserve_bytes": MIN_RUNNING_FREE, "threads": THREADS, "jobs": 1}


def _verify_reused_states() -> dict[str, Any]:
    if sha256_file(A02_FREEZE) != A02_FREEZE_SHA:
        raise BenchmarkError("A02 E79 trained freeze hash changed")
    freeze = _read_json(A02_FREEZE)
    if (freeze.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or freeze.get("train_rows") != 50_608 or freeze.get("train_cycles") != [47, 79]
            or freeze.get("train_target_sha256") != EXPECTED_TARGET_SHA
            or freeze.get("test_labels_loaded") is not False
            or freeze.get("test_or_sample_rows_loaded") is not False):
        raise BenchmarkError("A02 full-Train state population or access-boundary mismatch")
    manifest = _read_json(cgeom.MANIFEST_PATH)
    preflight = _read_json(cgeom.PREFLIGHT_PATH)
    a02_test_receipt = _read_json(cgeom.A02_TEST_RECEIPT)
    if (manifest.get("trained_freeze_sha256") != A02_FREEZE_SHA
            or manifest.get("source_commit") != freeze.get("source_commit")
            or manifest.get("test_target_loaded") is not False
            or manifest.get("used_test_for_model_selection") is not False
            or manifest.get("a02_bit_exact_replay") is not True
            or manifest.get("formula_audit", {}).get("status") != "PASS"
            or manifest.get("prediction_bundle_sha256") != PREDICTION_BUNDLE_SHA
            or sha256_file(PREDICTION_BUNDLE) != PREDICTION_BUNDLE_SHA
            or a02_test_receipt.get("npz_sha256") != sha256_file(A02_SOURCE_TEST_NPZ)
            or sha256_file(A02_SOURCE_TEST_NPZ) != A02_SOURCE_TEST_NPZ_SHA
            or preflight.get("fulltrain_feature_matrix_audit", {}).get("base_matrix_sha256") != BASE_MATRIX_SHA):
        raise BenchmarkError("A02/C2B integration manifests do not bind the expected frozen full-Train state")
    unit_ids = {
        "structured_state": cgeom.FIT_IDS[0], "scaler": cgeom.FIT_IDS[1],
        "ohe": cgeom.FIT_IDS[2], "c2b_lr": cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[3],
        "c1b_structured_state": cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[4],
        "c1b_hgb": cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[5],
    }
    receipts: dict[str, Any] = {}
    for role, fit_id in unit_ids.items():
        receipt = cgeom._verify_receipt(fit_id)
        receipts[role] = {"fit_id": fit_id, "artifact_sha256": receipt["artifact_sha256"],
                          "receipt_sha256": sha256_file(cgeom._unit_receipt_path(fit_id)),
                          "preparation_commit": receipt["preparation_commit"]}
    model = joblib.load(cgeom.A02_MODEL)
    model_sha = sha256_file(cgeom.A02_MODEL)
    vectorizer = model.text_space.vectorizer
    vectorizer_digest = cgeom.audit_c2b_vectorizer_compatibility(model=model)
    return {"a02_trained_freeze_sha256": A02_FREEZE_SHA,
            "a02_manifest_sha256": sha256_file(cgeom.MANIFEST_PATH),
            "c2b_preflight_sha256": sha256_file(cgeom.PREFLIGHT_PATH),
            "a02_model_sha256": model_sha,
            "vectorizer_vocabulary_size": len(vectorizer.vocabulary_),
            "vectorizer_compatibility": vectorizer_digest,
            "reused_units": receipts,
            "a02_original_test_predictions_sha256": sha256_file(A02_SOURCE_TEST_NPZ),
            "a02_original_test_receipt_sha256": sha256_file(cgeom.A02_TEST_RECEIPT),
            "a02_original_test_bundle_read": False,
            "frozen_candidate_probability_bundle_sha256": sha256_file(PREDICTION_BUNDLE),
            "candidate_probability_bundle_read": False,
            "train_target_sha256_from_freeze": freeze["train_target_sha256"]}


def run_synthetic() -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.models.classification_models import _TFIDF_PARAMS
    docs = ["alpha beta alpha", "beta gamma", "gamma alpha", "delta beta",
            "alpha delta", "beta beta gamma", "gamma delta", "alpha gamma"]
    query_docs = ["alpha beta", "gamma delta", "unseen omega", "", "café alpha"]
    vec = TfidfVectorizer(**{**_TFIDF_PARAMS, "min_df": 1, "max_df": 1.0, "max_features": 50})
    vec.fit(docs)
    w_train, _ = build_moment_matrix(docs, vec, shuffled=False)
    w_query_ref, _ = build_moment_matrix(query_docs, vec, shuffled=False)
    scalar_w_train = scalar_replay.rebuild_moments_scalar_order(docs, vec, shuffled=False)
    scalar_w_query = scalar_replay.rebuild_moments_scalar_order(query_docs, vec, shuffled=False)
    if sha256_csr(w_train) != sha256_csr(scalar_w_train) or sha256_csr(w_query_ref) != sha256_csr(scalar_w_query):
        raise BenchmarkError("Synthetic production/scalar REAL moments differ")
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    z0 = np.linspace(-0.25, 0.25, len(y), dtype=np.float64)
    with threadpool_limits(limits=THREADS):
        gamma, meta = fit_offset_coefficients(w_train, z0, y, c=1.0)
    with tempfile.TemporaryDirectory(prefix="pos-moments-e79-synthetic-") as temp:
        path = Path(temp) / "gamma.npz"
        synthetic_ids = np.asarray(["synthetic-é", "synthetic-日本"], dtype=str)
        _save_npz_once(path, gamma=gamma, record_ids=synthetic_ids)
        persisted = _load_npz(path)
        loaded = persisted["gamma"]
        if not np.array_equal(loaded, gamma):
            raise BenchmarkError("Safe numeric NPZ gamma persistence is not bit-exact")
        if not np.array_equal(persisted["record_ids"], synthetic_ids):
            raise BenchmarkError("Safe Unicode NPZ identity persistence is not exact")
        resumed = _load_npz(path)["gamma"]
        if not np.array_equal(resumed, gamma):
            raise BenchmarkError("Synthetic resume loaded a different durable gamma")
        p_present = expit(np.linspace(-0.6, 0.6, len(query_docs)) + scalar_w_query @ loaded)
        p_a02 = np.asarray([0.3, 0.7, 0.4, 0.9, 0.5], dtype=np.float64)
        p_missing = np.asarray([0.2, 0.4, 0.7, 0.9, 0.3], dtype=np.float64)
        missing = np.asarray([False, False, False, True, False])
        result = np.where(missing, p_missing, 0.5 * p_present + 0.5 * p_a02)
        expected = np.where(missing, p_missing, 0.5 * expit(np.linspace(-0.6, 0.6, len(query_docs)) + scalar_w_query @ gamma) + 0.5 * p_a02)
        if not np.array_equal(result, expected) or not np.array_equal(result[missing], p_missing[missing]):
            raise BenchmarkError("Synthetic present blend/missing fallback formula failed")
    return {"status": "PASS_SYNTHETIC_END_TO_END_AND_SAFE_RESUME",
            "synthetic_fit_only": True, "official_inputs_opened": False,
            "test_or_sample_opened": False, "target_scope": "synthetic only",
            "real_moment_builder_matches_independent_scalar_replay": True,
            "safe_unicode_id_roundtrip": True, "synthetic_optimizer_fit_count": 1,
            "gamma_persistence": "safe NPZ, allow_pickle=False, exact reload",
            "resume": "reused durable gamma without refitting",
            "fixed_policy_formula": "present=.5*expit(z0+W@gamma)+.5*p_A02; missing=pP exactly",
            "synthetic_fit_metadata": meta,
            "synthetic_gamma_sha256": sha256_array(gamma),
            "synthetic_train_W_sha256": sha256_csr(w_train),
            "synthetic_query_W_sha256": sha256_csr(scalar_w_query)}


def run_preflight() -> dict[str, Any]:
    if sha256_file(TRAIN) != TRAIN_SHA:
        raise BenchmarkError("Official Classification Train hash differs from frozen value")
    reused = _verify_reused_states()
    frame = cgeom.load_train_features()
    if "artifact_signal" in frame.columns:
        raise BenchmarkError("Feature-only Train reader exposed the target")
    cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    model = joblib.load(cgeom.A02_MODEL)
    vectorizer = model.text_space.vectorizer
    base, _, base_audit = cgeom._c2b_matrices(frame, vectorizer=vectorizer, train_side=False)
    if (base.shape != (50_608, 20_427) or base_audit["base_sha256"] != BASE_MATRIX_SHA
            or not np.isfinite(base.data).all()):
        raise BenchmarkError(f"Rebuilt C2_B base matrix differs from frozen integration: {base_audit}")
    documents, missing_from_text = cgeom.positional._text_values(frame.signal_stream)
    w_real = scalar_replay.rebuild_moments_scalar_order(documents, vectorizer, shuffled=False)
    w_builder, _ = build_moment_matrix(documents, vectorizer, shuffled=False)
    if (w_real.shape != (50_608, 17_259) or not np.isfinite(w_real.data).all()
            or sha256_csr(w_real) != sha256_csr(w_builder)
            or not np.array_equal(missing_from_text, np.asarray(base_audit["missing"], dtype=bool))):
        raise BenchmarkError("Canonical full-Train moments or missing-stream state failed independent preflight")
    del w_builder
    estimator = cgeom._load_active_c2b_lr()["estimator"]
    z0 = np.asarray(estimator.decision_function(base), dtype=np.float64).reshape(-1)
    if len(z0) != 50_608 or not np.isfinite(z0).all():
        raise BenchmarkError("Frozen C2_B full-Train base logits are invalid")
    if sha256_array(ids) != "9a96fa28e72b08a6022dd005aed877dadcc5e3dd5053200bcdd12711312c26b2":
        raise BenchmarkError("Official Train ID order differs from verified integration")
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BenchmarkError("Official Train cycles are not the complete 47-79 population")
    W_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if W_CACHE.exists() or ARRAY_CACHE.exists():
        raise BenchmarkError("Pre-fit feature cache already exists; refusing to overwrite")
    sparse.save_npz(W_CACHE, sparse.csr_matrix(w_real, dtype=np.float64), compressed=True)
    with ARRAY_CACHE.open("wb") as stream:
        np.savez_compressed(stream, record_ids=ids, cycles=cycles, missing_mask=missing_from_text,
                            base_logits=z0)
        stream.flush()
        os.fsync(stream.fileno())
    result = {
        "schema_version": 1, "status": "PASS_TRAIN_ONLY_TARGET_FREE_PREFIT",
        "benchmark_id": BENCHMARK_ID, "train_path": TRAIN.relative_to(ROOT).as_posix(),
        "train_sha256": TRAIN_SHA, "train_rows": 50_608, "train_cycles": [47, 79],
        "target_column_loaded": False, "target_sha256_computed": False,
        "test_or_sample_access": False, "ids_sha256": sha256_array(ids),
        "cycles_sha256": sha256_array(cycles), "missing_mask_sha256": sha256_array(missing_from_text),
        "base_matrix_shape": list(base.shape), "base_matrix_sha256": sha256_csr(base),
        "expected_base_matrix_sha256": BASE_MATRIX_SHA,
        "moment_matrix_shape": list(w_real.shape), "moment_matrix_nnz": int(w_real.nnz),
        "moment_matrix_sha256": sha256_csr(w_real),
        "moment_matrix_file_sha256": sha256_file(W_CACHE),
        "moments_builder": "V2 canonical scalar token-occurrence replay; bit-exactly matched the independent frozen POS_MOMENTS build_moment_matrix implementation",
        "base_logits_sha256": sha256_array(z0),
        "feature_cache_sha256": sha256_file(ARRAY_CACHE),
        "reused_state_audit": reused,
        "resource_snapshot_after_preflight": _resource(),
        "test_or_sample_opened": False,
    }
    _write_once(PREFIT_AUDIT, result)
    return result


def make_prefit_freeze() -> dict[str, Any]:
    _require_clean_tree()
    if not PREFIT_AUDIT.is_file() or not SYNTHETIC_QA.is_file():
        raise BenchmarkError("Synthetic QA and Train-only preflight must be generated and committed first")
    audit = _read_json(PREFIT_AUDIT)
    if audit.get("status") != "PASS_TRAIN_ONLY_TARGET_FREE_PREFIT" or sha256_file(TRAIN) != TRAIN_SHA:
        raise BenchmarkError("Feature-only preflight is absent, failed, or stale")
    source_commit = _git("rev-parse", "HEAD")
    source_hashes = _source_hashes()
    unsigned = {
        "schema_version": 1, "status": "FROZEN_BEFORE_FULLTRAIN_GAMMA_FIT",
        "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
        "source_commit": source_commit, "source_hashes": source_hashes,
        "recipe_sha256": sha256_file(REPORT / "RECIPE.json"),
        "authorization_sha256": sha256_file(REPORT / "BENCHMARK_AUTHORIZATION.md"),
        "synthetic_qa_sha256": sha256_file(SYNTHETIC_QA),
        "preflight_audit_sha256": sha256_file(PREFIT_AUDIT),
        "official_train_sha256": TRAIN_SHA, "official_train_rows": 50_608,
        "official_train_cycles": [47, 79], "fit_population_rows": 50_608,
        "fit_population_ids_sha256": audit["ids_sha256"],
        "fit_target_sha256_expected_from_a02_freeze": EXPECTED_TARGET_SHA,
        "feature_cache_sha256": audit["feature_cache_sha256"],
        "moment_matrix_file_sha256": audit["moment_matrix_file_sha256"],
        "moment_matrix_sha256": audit["moment_matrix_sha256"],
        "base_logits_sha256": audit["base_logits_sha256"],
        "reused_state_audit": audit["reused_state_audit"],
        "optimizer": {"name": "scipy.optimize.minimize:L-BFGS-B", "C": 1.0,
                      "initial_gamma": "zeros", "intercept": False,
                      "options": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}},
        "policy": {"present": ".5*expit(z0+W@gamma)+.5*p_A02",
                   "missing": "original full-Train C1_B HGB pP bit-exact", "threshold": 0.5},
        "test_features_or_probabilities_opened": False, "test_labels_opened": False,
        "sample_submission_opened": False, "kaggle_contacted": False,
    }
    freeze = {**unsigned, "freeze_sha256": _sha_obj(unsigned),
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(PREFIT_FREEZE, freeze)
    return freeze


def _ledger_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for i, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"Malformed live ledger line {i}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise BenchmarkError(f"Invalid live ledger event at line {i}")
        events.append(event)
    return events


def _snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    return cgeom._budget_snapshot(events)


def _append_ledger_lines(events: list[dict[str, Any]]) -> None:
    payload = b"".join(_canonical(e) + b"\n" for e in events)
    with LEDGER.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def reserve_fit() -> dict[str, Any]:
    _require_clean_tree()
    freeze = _read_json(PREFIT_FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_FULLTRAIN_GAMMA_FIT":
        raise BenchmarkError("Signed pre-fit freeze missing")
    if _sha_obj({k: v for k, v in freeze.items() if k not in {"freeze_sha256", "created_at_utc"}}) != freeze.get("freeze_sha256"):
        raise BenchmarkError("Pre-fit freeze signature mismatch")
    raw = LEDGER.read_bytes()
    live_sha = hashlib.sha256(raw).hexdigest()
    events = _ledger_events()
    snapshot = _snapshot(events)
    releases = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"
                and e.get("reservation_key") == RELEASE_KEY]
    if (len(events) != BASE_LEDGER_ROWS or live_sha != BASE_LEDGER_SHA
            or snapshot.get("global_cap") != OLD_GLOBAL_CAP
            or snapshot.get("category_caps") != OLD_CAPS
            or snapshot.get("active_reservations") != 137 or snapshot.get("fit_pending") != 0
            or len(releases) != 1 or _sha_obj(releases[0]) != RELEASE_EVENT_SHA):
        raise BenchmarkError(f"Live ledger differs from authorized reviewed checkpoint: {snapshot}")
    if any(e.get("reservation_key") == RESERVATION_KEY for e in events):
        raise BenchmarkError("Unique E79 reservation key already exists; no retry or duplicate reservation")
    if any(e.get("fit_id") == FIT_ID for e in events):
        raise BenchmarkError("Unique E79 fit ID already appears in ledger; no retry")
    cap_event = {
        "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
        "card_id": BENCHMARK_ID, "amendment_id": AMENDMENT_ID,
        "prior_global_cap": OLD_GLOBAL_CAP, "global_cap": NEW_GLOBAL_CAP,
        "prior_effective_caps": OLD_CAPS, "effective_caps": NEW_CAPS,
        "pre_amendment_ledger_sha256": live_sha, "pre_amendment_snapshot": snapshot,
        "freeze_signature": freeze["freeze_sha256"], "source_commit": _git("rev-parse", "HEAD"),
        "release_event_sha256": RELEASE_EVENT_SHA,
        "release_reservation_key": RELEASE_KEY,
        "reason": "One user-authorized exploratory POS_MOMENTS E79 gamma fit; prior released conditional E79 slot remains released and is not revived.",
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    reserve_event = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": BENCHMARK_ID, "stage": "REAL_GAMMA_FULLTRAIN",
        "reservation_key": RESERVATION_KEY, "category": "continuation", "fits": 1,
        "fit_ids": [FIT_ID], "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze["freeze_sha256"],
        "source_commit": _git("rev-parse", "HEAD"),
        "at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if hashlib.sha256(LEDGER.read_bytes()).hexdigest() != live_sha:
        raise BenchmarkError("Live shared ledger changed during reservation precheck")
    _append_ledger_lines([cap_event, reserve_event])
    post_raw = LEDGER.read_bytes()
    post_events = _ledger_events()
    post_snapshot = _snapshot(post_events)
    result = {"schema_version": 1, "status": "ONE_FIT_RESERVED",
              "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
              "baseline_ledger_sha256": live_sha, "baseline_event_rows": len(events),
              "prior_released_slot": {"reservation_key": RELEASE_KEY,
                                      "event_sha256": RELEASE_EVENT_SHA,
                                      "reused": False},
              "cap_amendment": cap_event, "reservation": reserve_event,
              "post_ledger_sha256": hashlib.sha256(post_raw).hexdigest(),
              "post_event_rows": len(post_events), "post_snapshot": post_snapshot,
              "no_test_or_sample_access": True}
    if (post_snapshot.get("global_cap") != NEW_GLOBAL_CAP
            or post_snapshot.get("category_caps") != NEW_CAPS
            or post_snapshot.get("active_reservations") != 138
            or post_snapshot.get("fit_pending") != 0):
        raise BenchmarkError(f"Post-reservation ledger invariants failed: {post_snapshot}")
    _write_once(RESERVATION_REPORT, result)
    return result


def _committed_file(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
    except subprocess.CalledProcessError:
        return False
    return hashlib.sha256(committed).hexdigest() == sha256_file(path)


def run_fulltrain_fit() -> dict[str, Any]:
    _require_clean_tree()
    freeze = _read_json(PREFIT_FREEZE)
    reservation = _read_json(RESERVATION_REPORT)
    if (not _committed_file(PREFIT_FREEZE) or not _committed_file(RESERVATION_REPORT)
            or reservation.get("status") != "ONE_FIT_RESERVED"
            or reservation.get("fit_id") != FIT_ID):
        raise BenchmarkError("Committed pre-fit freeze and one-fit reservation are required")
    _assert_sources(freeze["source_hashes"])
    events = _ledger_events()
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == FIT_ID]
    if starts or terminals or GAMMA_PATH.exists() or FIT_RECEIPT.exists() or TRAINED_FREEZE.exists():
        raise BenchmarkError("E79 fit state already exists; this authorization permits no retry")
    audit = _read_json(PREFIT_AUDIT)
    W = sparse.load_npz(W_CACHE).tocsr().astype(np.float64)
    arrays = _load_npz(ARRAY_CACHE)
    z0 = arrays["base_logits"].astype(np.float64)
    ids = arrays["record_ids"].astype(str)
    if (sha256_file(W_CACHE) != freeze["moment_matrix_file_sha256"]
            or sha256_csr(W) != freeze["moment_matrix_sha256"]
            or sha256_file(ARRAY_CACHE) != freeze["feature_cache_sha256"]
            or sha256_array(ids) != freeze["fit_population_ids_sha256"]):
        raise BenchmarkError("Committed feature cache no longer matches the pre-fit freeze")
    # This is the first read of Train target, after source and recipe freeze and reservation.
    target_frame = pd.read_csv(TRAIN, usecols=["artifact_signal"])
    y = pd.to_numeric(target_frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    if (len(y) != 50_608 or not np.array_equal(np.unique(y), np.asarray([0, 1], dtype=np.int8))
            or sha256_array(y) != EXPECTED_TARGET_SHA):
        raise BenchmarkError("Train labels differ from the verified A02 target population")
    target_sha = sha256_array(y)
    resource = _assert_resource(initial=True, estimated_incremental=max(1 * 1024**3, 4 * (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) + 256 * 1024**2))
    started = {"event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
               "card_id": BENCHMARK_ID, "stage": "REAL_GAMMA_FULLTRAIN",
               "reservation_key": RESERVATION_KEY, "amendment_id": AMENDMENT_ID,
               "fit_id": FIT_ID, "fit_kind": "PREDICTOR", "fit_consumed_at_start": True,
               "fit_population": {"rows": len(y), "cycles": [47, 79],
                                  "record_ids_sha256": sha256_array(ids),
                                  "target_sha256": target_sha, "target_read_purpose": "fulltrain_gamma_fit_only",
                                  "moment_matrix_sha256": sha256_csr(W), "base_logits_sha256": sha256_array(z0)},
               "resource_snapshot": resource, "source_commit": freeze["source_commit"],
               "freeze_signature": freeze["freeze_sha256"], "at_utc": datetime.now(timezone.utc).isoformat()}
    _append_ledger_lines([started])
    try:
        with threadpool_limits(limits=THREADS):
            gamma, meta = fit_offset_coefficients(W, z0, y, c=1.0)
        if not meta.get("success") or not np.isfinite(gamma).all():
            raise BenchmarkError(f"One-shot gamma fit did not satisfy the frozen optimizer contract: {meta}")
        _save_npz_once(GAMMA_PATH, gamma=gamma)
        fit_receipt = {"schema_version": 1, "status": "PASS_ONE_FULLTRAIN_GAMMA_FIT_SAVED",
                       "fit_id": FIT_ID, "benchmark_id": BENCHMARK_ID,
                       "source_commit": freeze["source_commit"],
                       "source_hashes": freeze["source_hashes"],
                       "train_sha256": TRAIN_SHA, "fit_rows": len(y),
                       "fit_cycles": [47, 79], "fit_ids_sha256": sha256_array(ids),
                       "target_sha256": target_sha,
                       "moment_matrix_sha256": sha256_csr(W),
                       "base_logits_sha256": sha256_array(z0),
                       "gamma_sha256": sha256_array(gamma), "gamma_file_sha256": sha256_file(GAMMA_PATH),
                       "optimizer": meta, "resource_before_fit": resource,
                       "fit_count_for_benchmark": 1, "retry_count": 0,
                       "test_features_or_probabilities_opened": False,
                       "test_labels_opened": False, "sample_submission_opened": False,
                       "kaggle_contacted": False, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_once(FIT_RECEIPT, fit_receipt)
        trained_unsigned = {"schema_version": 1, "status": "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE",
                            "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
                            "source_commit": freeze["source_commit"],
                            "pre_fit_freeze_sha256": sha256_file(PREFIT_FREEZE),
                            "fit_receipt_sha256": sha256_file(FIT_RECEIPT),
                            "gamma_sha256": sha256_array(gamma),
                            "gamma_file_sha256": sha256_file(GAMMA_PATH),
                            "train_sha256": TRAIN_SHA, "fit_rows": 50_608,
                            "target_sha256": target_sha, "fit_population_cycles": [47, 79],
                            "test_or_sample_rows_loaded": False, "test_labels_loaded": False,
                            "inference_started": False, "kaggle_contacted": False}
        trained = {**trained_unsigned, "freeze_sha256": _sha_obj(trained_unsigned),
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_once(TRAINED_FREEZE, trained)
        saved_event = {"event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
                       "card_id": BENCHMARK_ID, "stage": "REAL_GAMMA_FULLTRAIN",
                       "reservation_key": RESERVATION_KEY, "amendment_id": AMENDMENT_ID,
                       "fit_id": FIT_ID, "fit_kind": "PREDICTOR",
                       "receipt_sha256": sha256_file(FIT_RECEIPT),
                       "source_commit": freeze["source_commit"],
                       "freeze_signature": freeze["freeze_sha256"],
                       "at_utc": datetime.now(timezone.utc).isoformat()}
        _append_ledger_lines([saved_event])
        del target_frame, W, arrays, gamma, y, z0
        return {"status": "PASS_ONE_FIT_AND_TRAINED_STATE_FROZEN", "fit_receipt": fit_receipt,
                "trained_freeze": trained, "ledger_fit_saved": True}
    except Exception as exc:
        current = _ledger_events()
        has_terminal = any(e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == FIT_ID for e in current)
        if not has_terminal and not (GAMMA_PATH.exists() or FIT_RECEIPT.exists() or TRAINED_FREEZE.exists()):
            _append_ledger_lines([{"event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                                   "card_id": BENCHMARK_ID, "stage": "REAL_GAMMA_FULLTRAIN",
                                   "reservation_key": RESERVATION_KEY, "amendment_id": AMENDMENT_ID,
                                   "fit_id": FIT_ID, "fit_kind": "PREDICTOR",
                                   "reason": f"One-shot fit failed: {type(exc).__name__}: {exc}",
                                   "source_commit": freeze["source_commit"],
                                   "freeze_signature": freeze["freeze_sha256"],
                                   "at_utc": datetime.now(timezone.utc).isoformat()}])
        raise


def _verify_trained_freeze() -> dict[str, Any]:
    _require_clean_tree()
    if not _committed_file(TRAINED_FREEZE) or not _committed_file(FIT_RECEIPT):
        raise BenchmarkError("Trained-state freeze and fit receipt must be committed before Test access")
    freeze = _read_json(TRAINED_FREEZE)
    receipt = _read_json(FIT_RECEIPT)
    unsigned = {k: v for k, v in freeze.items() if k not in {"freeze_sha256", "created_at_utc"}}
    if (freeze.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or freeze.get("benchmark_id") != BENCHMARK_ID or freeze.get("fit_id") != FIT_ID
            or _sha_obj(unsigned) != freeze.get("freeze_sha256")
            or freeze.get("test_labels_loaded") is not False
            or freeze.get("test_or_sample_rows_loaded") is not False
            or receipt.get("status") != "PASS_ONE_FULLTRAIN_GAMMA_FIT_SAVED"
            or receipt.get("fit_id") != FIT_ID or receipt.get("train_sha256") != TRAIN_SHA
            or receipt.get("target_sha256") != EXPECTED_TARGET_SHA
            or receipt.get("retry_count") != 0
            or sha256_file(GAMMA_PATH) != freeze.get("gamma_file_sha256")
            or sha256_file(FIT_RECEIPT) != freeze.get("fit_receipt_sha256")
            or sha256_file(PREFIT_FREEZE) != freeze.get("pre_fit_freeze_sha256")):
        raise BenchmarkError("Committed trained-state freeze/receipt integrity mismatch")
    gamma = _load_npz(GAMMA_PATH)["gamma"]
    if sha256_array(gamma) != freeze.get("gamma_sha256"):
        raise BenchmarkError("Gamma coefficient hash differs from trained-state freeze")
    return freeze


def run_inference() -> dict[str, Any]:
    trained = _verify_trained_freeze()
    if sha256_file(TEST) != TEST_SHA:
        raise BenchmarkError("Official Classification Test hash differs from frozen input")
    test_columns = list(dict.fromkeys(["record_id", "cycle_code", *cgeom.CLASSIFICATION_FEATURES]))
    test = pd.read_csv(TEST, usecols=test_columns,
                       dtype={"record_id": str})
    if "artifact_signal" in test.columns or len(test) != 29_421:
        raise BenchmarkError("Test feature reader loaded a target or wrong row count")
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (test.record_id.isna().any() or (ids == "").any() or test.record_id.duplicated().any()
            or not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64))):
        raise BenchmarkError("Official Test IDs/cycle support failed")
    if (sha256_file(A02_SOURCE_TEST_NPZ) != A02_SOURCE_TEST_NPZ_SHA
            or sha256_file(PREDICTION_BUNDLE) != PREDICTION_BUNDLE_SHA):
        raise BenchmarkError("Verified A02/C2B Test prediction bundle hash changed")
    base, _, base_audit = cgeom._c2b_matrices(test, train_side=False)
    estimator = cgeom._load_active_c2b_lr()["estimator"]
    z0 = np.asarray(estimator.decision_function(base), dtype=np.float64).reshape(-1)
    vectorizer = joblib.load(cgeom.A02_MODEL).text_space.vectorizer
    documents, missing = cgeom.positional._text_values(test.signal_stream)
    W = scalar_replay.rebuild_moments_scalar_order(documents, vectorizer, shuffled=False)
    gamma = _load_npz(GAMMA_PATH)["gamma"]
    if (W.shape[1] != len(gamma) or len(z0) != len(test)
            or not np.array_equal(missing, np.asarray(base_audit["missing"], dtype=bool))):
        raise BenchmarkError("Test C2_B/moments/gamma dimensions or availability disagree")
    with np.load(PREDICTION_BUNDLE, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    with np.load(A02_SOURCE_TEST_NPZ, allow_pickle=False) as archive:
        original_a02 = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"record_ids", "missing_mask", "a02_probability", "c1b_missing_probability", "posoffset_probability"}
    if not required.issubset(arrays):
        raise BenchmarkError(f"A02 bundle lacks required integrity-only arrays: {sorted(required-set(arrays))}")
    if not {"record_ids", "candidate_probability"}.issubset(original_a02):
        raise BenchmarkError("Original A02 prediction bundle lacks its ID/probability vectors")
    if (not np.array_equal(ids, arrays["record_ids"].astype(str))
            or not np.array_equal(ids, original_a02["record_ids"].astype(str))
            or not np.array_equal(arrays["a02_probability"], original_a02["candidate_probability"])
            or not np.array_equal(missing, arrays["missing_mask"].astype(bool))):
        raise BenchmarkError("A02 bundle probability/IDs or availability mask do not match official Test features")
    p_a02 = arrays["a02_probability"].astype(np.float64)
    p_p = arrays["c1b_missing_probability"].astype(np.float64)
    p_pos = arrays["posoffset_probability"].astype(np.float64)
    if any(value.shape != (len(ids),) or not np.isfinite(value).all()
           for value in (p_a02, p_p, p_pos)):
        raise BenchmarkError("Original A02 prediction bundle contains an invalid probability vector")
    q = np.asarray(expit(z0 + W @ gamma), dtype=np.float64).reshape(-1)
    p = np.where(missing, p_p, 0.5 * q + 0.5 * p_a02)
    labels = (p >= 0.50).astype(np.int8)
    # Comparators are reproduced only for identity, route, and duplicate-vector checks.
    expected_pos = np.where(missing, p_p, p_pos)
    expected_mix50 = np.where(missing, p_p, 0.5 * p_pos + 0.5 * p_a02)
    labels_a02 = (p_a02 >= 0.50).astype(np.int8)
    labels_pos = (expected_pos >= 0.50).astype(np.int8)
    labels_mix50 = (expected_mix50 >= 0.50).astype(np.int8)
    if (not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.array_equal(p[missing], p_p[missing])
            or labels.dtype != np.int8 or not np.isin(labels, [0, 1]).all()):
        raise BenchmarkError("E79 fixed policy generated invalid probabilities or labels")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        raise BenchmarkError("Refusing to overwrite the one authorized candidate CSV")
    pd.DataFrame({"record_id": ids, "artifact_signal": labels}).to_csv(CSV, index=False, lineterminator="\n")
    comparisons = {"A02": sha256_array(labels_a02), "POS_BAG_OFFSET": sha256_array(labels_pos),
                   "MIX50": sha256_array(labels_mix50), "E79": sha256_array(labels)}
    qa = {"schema_version": 1, "status": "INFERENCE_BUILT_PENDING_INDEPENDENT_VERIFICATION",
          "benchmark_id": BENCHMARK_ID, "csv_path": CSV.relative_to(ROOT).as_posix(),
          "csv_sha256": sha256_file(CSV), "test_sha256": TEST_SHA,
          "a02_original_test_predictions_sha256": A02_SOURCE_TEST_NPZ_SHA,
          "frozen_candidate_probability_bundle_sha256": PREDICTION_BUNDLE_SHA,
          "trained_state_freeze_sha256": sha256_file(TRAINED_FREEZE),
          "rows": len(ids), "columns": ["record_id", "artifact_signal"],
          "ordered_test_ids_sha256": sha256_array(ids), "cycle_counts": pd.Series(cycles).value_counts().sort_index().to_dict(),
          "missing_rows": int(missing.sum()), "present_rows": int((~missing).sum()),
          "base_matrix_sha256": sha256_csr(base), "moment_matrix_sha256": sha256_csr(W),
          "base_logits_sha256": sha256_array(z0), "gamma_sha256": sha256_array(gamma),
          "present_probability_sha256": sha256_array(q), "candidate_probability_sha256": sha256_array(p),
          "candidate_label_sha256": sha256_array(labels),
          "label_identity_hashes_for_duplicate_check_only": comparisons,
          "unique_label_vectors": len(set(comparisons.values())),
          "threshold": 0.5, "test_labels_loaded": False,
          "sample_submission_opened": False, "kaggle_contacted": False,
          "trained_freeze": trained, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(INFERENCE_BUILD, qa)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("synthetic", "preflight", "freeze", "reserve", "fit", "infer", "verify"))
    args = parser.parse_args()
    if args.action == "synthetic":
        result = run_synthetic()
        _write_once(SYNTHETIC_QA, result)
    elif args.action == "preflight":
        result = run_preflight()
    elif args.action == "freeze":
        result = make_prefit_freeze()
    elif args.action == "reserve":
        result = reserve_fit()
    elif args.action == "fit":
        result = run_fulltrain_fit()
    elif args.action == "infer":
        result = run_inference()
    else:
        from scripts.verify_optimization_v3_positional_moments_real_e79_benchmark_v1 import verify_outputs
        result = verify_outputs()
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
