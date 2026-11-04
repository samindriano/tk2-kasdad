"""One-shot full-Train REAL POS_CONTRAST E79 benchmark.

This is a new exploratory namespace. It reuses the verified full-Train C2_B,
TF-IDF, A02, and C1_B states; fits exactly one contrast theta on official Train;
then builds one local Test-inference CSV. It never reads Test labels or uploads.
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
from scripts import optimization_v3_positional_moments_real_e79_benchmark_v1 as moments
from scripts import optimization_v3_positional_text_pilot_v1 as positional
from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
from src.models.positional_bag_offset_v1 import fit_offset_coefficients
from src.models.positional_contrast_v1 import contrast_block
from src.models.positional_moments_v1 import sha256_array, sha256_csr, sha256_file


BENCHMARK_ID = "POS_CONTRAST_REAL_E79_EXPLORATORY_V1"
FIT_ID = f"{BENCHMARK_ID}_THETA_FULLTRAIN"
RESERVATION_KEY = f"{BENCHMARK_ID}:ONE_THETA_FULLTRAIN"
TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
TEST_SHA = "4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c"
SAMPLE_SHA = "889f97d69659a64945421d841dcf96e5813020606f29b2992d4b9560cecdcba7"
A02_FREEZE_SHA = "ab0307e6e1c9778ed0f091a3d6ed55e0ac8705ae31b9cb29b9f083e46b5c2b82"
A02_SOURCE_TEST_NPZ_SHA = "c267a4b998aca64982b6f63b10bedb2f3b3061007166db7731d221788ac89e47"
PREDICTION_BUNDLE_SHA = "ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271"
BASE_MATRIX_SHA = "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b"
EXPECTED_IDS_SHA = "9a96fa28e72b08a6022dd005aed877dadcc5e3dd5053200bcdd12711312c26b2"
OLD_RELEASE_KEY = "POS_CONTRAST_V1:CONDITIONAL_E79_AFTER_SOL_REVIEW"
LEDGER = cgeom.LEDGER
REPORT = ROOT / "reports/optimization_v3/contrast_real_e79_exploratory_v1"
RUN = ROOT / "experiments/runs/optimization_v3_contrast_real_e79_exploratory_v1"
SUBMISSIONS = ROOT / "submissions/optimization_v3/contrast_real_e79_exploratory_v1"
TRAIN = ROOT / "data/raw/classification/Train.csv"
TEST = ROOT / "data/raw/classification/Test.csv"
SAMPLE = ROOT / "data/raw/classification/SampleSubmission.csv"
CSV = SUBMISSIONS / "classification_pos_contrast_real_e79_exploratory.csv"
Z_TRAIN_PATH = RUN / "prefit" / "contrast_train.npz"
TRAIN_META_PATH = RUN / "prefit" / "train_meta.npz"
THETA_PATH = RUN / "fit" / "theta.npz"
PREDICTION_PATH = RUN / "inference" / "predictions.npz"

RECIPE = REPORT / "RECIPE.json"
AUTHORIZATION = REPORT / "BENCHMARK_AUTHORIZATION.md"
SYNTHETIC_QA = REPORT / "SYNTHETIC_QA.json"
PREFIT_AUDIT = REPORT / "PREFIT_AUDIT.json"
PREFIT_FREEZE = REPORT / "PRE_FIT_FREEZE_R3.json"
RESERVATION_REPORT = REPORT / "RESERVATION.json"
FIT_RECEIPT = REPORT / "FIT_RECEIPT.json"
TRAINED_FREEZE = REPORT / "TRAINED_STATE_FREEZE.json"
INFERENCE_BUILD = REPORT / "INFERENCE_BUILD.json"
QA_REPORT = REPORT / "INFERENCE_QA.json"
MANIFEST = REPORT / "BENCHMARK_MANIFEST.json"
RESULTS_MD = REPORT / "RESULTS.md"

MIN_INITIAL_FREE = 4 * 1024**3
MIN_RUNNING_FREE = 2 * 1024**3
MAX_RSS = 6 * 1024**3
THREADS = 2

SOURCE_PATHS = (
    "scripts/optimization_v3_positional_contrast_real_e79_exploratory_v1.py",
    "scripts/verify_optimization_v3_positional_contrast_real_e79_exploratory_v1.py",
    "src/models/positional_contrast_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "src/models/positional_moments_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/optimization_v3_a02_cgeom_benchmark_v1.py",
    "scripts/optimization_v3_positional_moments_real_e79_benchmark_v1.py",
    "reports/optimization_v3/positional_contrast_v1/RECIPE.json",
    "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_pilot_v1/RECIPE.json",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/E79_TRAINED_FREEZE.json",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/FROZEN_RECIPES.json",
    "reports/optimization_v3/a02_cgeom_benchmark_v1/BATCH_MANIFEST.json",
    "experiments/runs/optimization_v3_a02_cgeom_benchmark_v1/INTEGRATION_PREFLIGHT_R1.json",
    "reports/optimization_v3/a02_continuation_v1/submission_manifest.json",
    "reports/optimization_v3/positional_moments_real_e79_benchmark_v1/BENCHMARK_MANIFEST.json",
    "reports/optimization_v3/bigram_position_real_e79_v1/RECIPE.json",
    "reports/optimization_v3/contrast_real_e79_exploratory_v1/RECIPE.json",
    "reports/optimization_v3/contrast_real_e79_exploratory_v1/BENCHMARK_AUTHORIZATION.md",
    "tests/test_optimization_v3_positional_contrast_real_e79_exploratory_v1.py",
)


class BenchmarkError(RuntimeError):
    """The one-shot benchmark failed a frozen integrity or resource gate."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                         allow_nan=False, indent=2).encode("utf-8") + b"\n"
    _write_bytes_once(path, payload)


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise BenchmarkError(f"Refusing to replace immutable output: {path}")
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
    except FileExistsError as exc:
        raise BenchmarkError(f"Output appeared concurrently: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _save_npz_once(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BenchmarkError(f"Refusing to replace NPZ output: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
    except FileExistsError as exc:
        raise BenchmarkError(f"NPZ output appeared concurrently: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for rel in SOURCE_PATHS:
        path = ROOT / rel
        if not path.is_file():
            raise BenchmarkError(f"Frozen source is missing: {rel}")
        result[rel] = sha256_file(path)
    return result


def _assert_sources(expected: Mapping[str, str]) -> None:
    if _source_hashes() != dict(expected):
        raise BenchmarkError("Frozen source, recipe, or tests changed after pre-fit freeze")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _require_clean_tree() -> None:
    if _git("status", "--porcelain"):
        raise BenchmarkError("Commit the exact current preparation/report files before this phase")
    if _git("branch", "--show-current") != "main":
        raise BenchmarkError("Authorized worktree must remain on main")


def _committed_file(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
    except subprocess.CalledProcessError:
        return False
    return hashlib.sha256(committed).hexdigest() == sha256_file(path)


def _ledger_events() -> list[dict[str, Any]]:
    return moments._ledger_events()


def _ledger_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    return cgeom._budget_snapshot(events)


def _category_caps(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    return snapshot.get("category_caps", snapshot.get("effective_category_caps", {}))


def _append_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = {**_jsonable(event), "at_utc": datetime.now(timezone.utc).isoformat()}
    with LEDGER.open("ab") as stream:
        stream.write(_canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _load_npz(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if keys is None else keys
        missing = set(selected) - set(archive.files)
        if missing:
            raise BenchmarkError(f"NPZ lacks required arrays {sorted(missing)}: {path}")
        return {key: np.asarray(archive[key]).copy() for key in selected}


def _resource_state() -> dict[str, int]:
    process = psutil.Process(os.getpid())
    return {"process_rss_bytes": int(process.memory_info().rss),
            "available_ram_bytes": int(psutil.virtual_memory().available)}


def _classify_python_process(name: str, cmdline: str, rss_bytes: int) -> str:
    """Treat only small, unmarked pythonw background services as non-training."""
    lower = f"{name} {cmdline}".lower()
    training_markers = ("optimization_v3", "tk2-kasdad", "jupyter", "ipykernel",
                        "notebook", "pytest", "xgboost", "lightgbm", "sklearn",
                        "train.py", "fit.py", "run_fit")
    if any(marker in lower for marker in training_markers):
        return "potential_training"
    if name.lower().startswith("pythonw") and rss_bytes <= 256 * 1024**2:
        return "lightweight_pythonw_background"
    return "unclassified_python_process"


def _python_process_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    tolerated: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if int(proc.info["pid"] or -1) == os.getpid():
                continue
            name = str(proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])[:800]
            rss = int(proc.memory_info().rss)
            item = {"pid": int(proc.info["pid"]), "name": name,
                    "cmdline": cmdline, "rss_bytes": rss}
            classification = _classify_python_process(name, cmdline, rss)
            item["classification"] = classification
            if classification == "lightweight_pythonw_background":
                tolerated.append(item)
            else:
                blockers.append(item)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return blockers, tolerated


def _resource_gate(*, initial: bool, estimated_incremental: int = 0) -> dict[str, Any]:
    state = _resource_state()
    blockers, tolerated = _python_process_inventory()
    free_min = MIN_INITIAL_FREE if initial else MIN_RUNNING_FREE
    projected_free = state["available_ram_bytes"] - estimated_incremental
    projected_rss = state["process_rss_bytes"] + estimated_incremental
    if blockers:
        raise BenchmarkError(f"Possible overlapping Python/training process: {blockers}")
    if (state["available_ram_bytes"] < free_min
            or projected_free < MIN_RUNNING_FREE
            or projected_rss > MAX_RSS):
        raise BenchmarkError(f"RAM/resource gate failed: state={state}, estimate={estimated_incremental}")
    return {**state, "possible_training_processes": blockers,
            "lightweight_background_pythonw": tolerated,
            "estimated_incremental_bytes": int(estimated_incremental),
            "projected_available_ram_bytes": int(projected_free),
            "projected_process_rss_bytes": int(projected_rss),
            "minimum_initial_free_bytes": MIN_INITIAL_FREE,
            "minimum_running_reserve_bytes": MIN_RUNNING_FREE,
            "rss_ceiling_bytes": MAX_RSS, "threads": THREADS, "jobs": 1}


def _verify_reused_states() -> dict[str, Any]:
    if sha256_file(TRAIN) != TRAIN_SHA:
        raise BenchmarkError("Official Classification Train SHA mismatch")
    if sha256_file(cgeom.TRAINED_FREEZE) != A02_FREEZE_SHA:
        raise BenchmarkError("A02 E79 trained-state freeze changed")
    reused = moments._verify_reused_states()
    freeze = _read_json(cgeom.TRAINED_FREEZE)
    manifest = _read_json(cgeom.MANIFEST_PATH)
    recipes = _read_json(cgeom.RECIPE_PATH)
    if (freeze.get("status") != "TRAINED_STATE_FROZEN_BEFORE_TEST_INFERENCE"
            or freeze.get("train_rows") != 50_608 or freeze.get("train_cycles") != [47, 79]
            or freeze.get("test_labels_loaded") is not False
            or manifest.get("test_target_loaded") is not False
            or manifest.get("used_test_for_model_selection") is not False):
        raise BenchmarkError("Existing full-Train A02 source boundaries failed")
    c2b_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[3]
    c1b_id = cgeom.INTEGRATION_ACTIVE_ID_BY_SLOT[5]
    c2b = cgeom._verify_receipt(c2b_id)
    c1b = cgeom._verify_receipt(c1b_id)
    for name, receipt in (("C2B", c2b), ("C1B", c1b)):
        pop = receipt.get("fit_population", {})
        if (pop.get("rows") != 50_608 or pop.get("cycles") != [47, 79]
                or pop.get("source") != "official Classification Train"):
            raise BenchmarkError(f"Reusable {name} fit population does not match official full Train")
    return {"a02_freeze_sha256": sha256_file(cgeom.TRAINED_FREEZE),
            "a02_model_sha256": sha256_file(cgeom.A02_MODEL),
            "a02_fit_receipt_sha256": sha256_file(cgeom.A02_FIT_RECEIPT),
            "a02_fit_state_sha256": sha256_file(cgeom.A02_STATE),
            "a02_test_prediction_sha256": sha256_file(cgeom.A02_TEST_NPZ),
            "a02_test_prediction_receipt_sha256": sha256_file(cgeom.A02_TEST_RECEIPT),
            "a02_prediction_bundle_sha256": sha256_file(cgeom.OUTPUT_NPZ),
            "a02_manifest_sha256": sha256_file(cgeom.MANIFEST_PATH),
            "a02_recipes_sha256": sha256_file(cgeom.RECIPE_PATH),
            "fulltrain_preflight_sha256": sha256_file(cgeom.PREFLIGHT_PATH),
            "c2b_fit_id": c2b_id, "c2b_artifact_sha256": c2b["artifact_sha256"],
            "c2b_receipt_sha256": sha256_file(cgeom._unit_receipt_path(c2b_id)),
            "c1b_fit_id": c1b_id, "c1b_artifact_sha256": c1b["artifact_sha256"],
            "c1b_receipt_sha256": sha256_file(cgeom._unit_receipt_path(c1b_id)),
            "reused_moments_audit": reused,
            "reused_components_fitted_again": False}


def _build_contrast(frame: pd.DataFrame, *, vectorizer: Any) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
    if "artifact_signal" in frame.columns:
        raise BenchmarkError("Feature reader exposed the target")
    base, _unused_position, base_audit = cgeom._c2b_matrices(
        frame, vectorizer=vectorizer, train_side=False)
    docs, missing = cgeom.positional._text_values(frame.signal_stream)
    position, position_audit = build_position_block(docs, vectorizer, shuffled=False)
    augmented = sparse.hstack((base, position), format="csr", dtype=np.float64)
    projected = contrast_block(augmented, base.shape[1])
    estimator = cgeom._load_active_c2b_lr()["estimator"]
    logits = np.asarray(estimator.decision_function(base), dtype=np.float64).reshape(-1)
    # Verify the frozen C2_B offset definition directly: intercept + base @ coef.
    direct = np.asarray(estimator.intercept_[0] + base @ estimator.coef_[0], dtype=np.float64).reshape(-1)
    if (not np.isfinite(projected.data).all() or not np.isfinite(logits).all()
            or not np.array_equal(missing, np.asarray(base_audit["missing"], dtype=bool))
            or not np.allclose(logits, direct, rtol=0, atol=1e-12)):
        raise BenchmarkError("C2_B offset or REAL contrast feature construction failed")
    base_audit_summary = {key: value for key, value in base_audit.items() if key != "missing"}
    base_audit_summary["missing_rows"] = int(np.count_nonzero(missing))
    base_audit_summary["missing_mask_sha256"] = sha256_array(missing)
    audit = {"base_shape": list(base.shape), "base_sha256": sha256_csr(base),
             "base_audit": base_audit_summary, "position_shape": list(position.shape),
             "position_sha256": sha256_csr(position), "position_builder_audit": position_audit,
             "augmented_shape": list(augmented.shape), "augmented_sha256": sha256_csr(augmented),
             "contrast_shape": list(projected.shape), "contrast_sha256": sha256_csr(projected),
             "contrast_nnz": int(projected.nnz), "base_logits_sha256": sha256_array(logits),
             "missing_mask_sha256": sha256_array(missing)}
    del base, position, augmented
    return projected.tocsr(), logits, missing, audit


def run_synthetic() -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.models.classification_models import _TFIDF_PARAMS

    docs = ["alpha beta alpha", "beta gamma", "gamma alpha", "delta beta",
            "alpha delta", "beta beta gamma", "gamma delta", "alpha gamma"]
    query_docs = ["alpha beta", "gamma delta", "unseen omega", "", "cafe alpha"]
    vectorizer = TfidfVectorizer(**{**_TFIDF_PARAMS, "min_df": 1, "max_df": 1.0,
                                   "max_features": 50})
    bag = vectorizer.fit_transform(docs).tocsr().astype(np.float64)
    pos, _ = build_position_block(docs, vectorizer, shuffled=False)
    augmented = sparse.hstack((bag, pos), format="csr", dtype=np.float64)
    z = contrast_block(augmented, bag.shape[1])
    expected_first = pos[:, :pos.shape[1] // 3]
    expected_middle = pos[:, pos.shape[1] // 3:2 * (pos.shape[1] // 3)]
    expected_last = pos[:, 2 * (pos.shape[1] // 3):]
    expected_z = sparse.hstack(((expected_first - expected_last) / np.sqrt(2.0),
                                (expected_first - 2 * expected_middle + expected_last) / np.sqrt(6.0)),
                               format="csr")
    if sha256_csr(z) != sha256_csr(expected_z):
        raise BenchmarkError("Synthetic exact positional contrast formula mismatch")
    target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    base_logits = np.linspace(-0.2, 0.2, len(target), dtype=np.float64)
    with threadpool_limits(limits=THREADS):
        theta, optimizer = fit_offset_coefficients(z, base_logits, target, c=1.0)
    query_bag = vectorizer.transform(query_docs).tocsr().astype(np.float64)
    query_pos, _ = build_position_block(query_docs, vectorizer, shuffled=False)
    query_z = contrast_block(sparse.hstack((query_bag, query_pos), format="csr"), query_bag.shape[1])
    with tempfile.TemporaryDirectory(prefix="pos-contrast-e79-synthetic-") as tmp:
        path = Path(tmp) / "theta.npz"
        test_ids = np.asarray(["synthetic-é", "synthetic-日本"], dtype=str)
        _save_npz_once(path, theta=theta, record_ids=test_ids)
        loaded = _load_npz(path)
        if not np.array_equal(loaded["theta"], theta) or not np.array_equal(loaded["record_ids"], test_ids):
            raise BenchmarkError("Synthetic theta/Unicode identity persistence is not exact")
        resumed = _load_npz(path)["theta"]
        if not np.array_equal(resumed, theta):
            raise BenchmarkError("Synthetic safe-resume did not load the durable theta")
    q = expit(np.linspace(-0.3, 0.3, len(query_docs)) + query_z @ theta)
    q0 = expit(np.linspace(-0.3, 0.3, len(query_docs)))
    p_a02 = np.asarray([0.3, 0.7, 0.4, 0.9, 0.5], dtype=np.float64)
    p_c1b = np.asarray([0.2, 0.4, 0.7, 0.9, 0.3], dtype=np.float64)
    missing = np.asarray([False, False, False, True, False])
    p = np.where(missing, p_c1b, 0.5 * q + 0.5 * p_a02)
    p0 = np.where(missing, p_c1b, 0.5 * q0 + 0.5 * p_a02)
    if (not np.array_equal(p[missing], p_c1b[missing])
            or not np.isfinite(p).all() or not np.isfinite(p0).all()
            or not np.isin((p >= 0.5).astype(np.int8), [0, 1]).all()):
        raise BenchmarkError("Synthetic fixed policy/gamma-zero replay failed")
    return {"schema_version": 1, "status": "PASS_SYNTHETIC_END_TO_END_AND_SAFE_RESUME",
            "official_inputs_opened": False, "test_or_sample_opened": False,
            "synthetic_gamma_fit_count": 1, "official_fit_count": 0,
            "exact_contrast_formula": True, "no_post_projection_normalization": True,
            "optimizer": optimizer, "theta_sha256": sha256_array(theta),
            "train_contrast_sha256": sha256_csr(z), "query_contrast_sha256": sha256_csr(query_z),
            "gamma_zero_replay": "PASS", "missing_fallback_exact": True,
            "persistence_and_resume": "PASS", "kaggle_contacted": False}


def run_preflight() -> dict[str, Any]:
    if sha256_file(TRAIN) != TRAIN_SHA:
        raise BenchmarkError("Official Train hash does not match frozen source")
    reused = _verify_reused_states()
    resource = _resource_gate(initial=True, estimated_incremental=1024**3)
    frame = cgeom.load_train_features()
    if "artifact_signal" in frame.columns or len(frame) != 50_608:
        raise BenchmarkError("Target-free official Train feature reader has invalid schema/population")
    ids = frame.record_id.astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (sha256_array(ids) != EXPECTED_IDS_SHA
            or not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64))):
        raise BenchmarkError("Official Train ID order or cycle population differs from frozen full-Train state")
    a02_model = joblib.load(cgeom.A02_MODEL)
    vectorizer = a02_model.text_space.vectorizer
    W, z0, missing, matrix_audit = _build_contrast(frame, vectorizer=vectorizer)
    if matrix_audit["base_sha256"] != BASE_MATRIX_SHA:
        raise BenchmarkError("Rebuilt C2_B full-Train base differs from frozen integration matrix")
    expected_contrast_width = 2 * int(matrix_audit["position_builder_audit"]["unigram_vocabulary_size"])
    if (W.shape != (50_608, expected_contrast_width)
            or not np.isfinite(W.data).all()):
        raise BenchmarkError(f"Full-Train contrast matrix shape/data is invalid: {W.shape}")
    if Z_TRAIN_PATH.exists() or TRAIN_META_PATH.exists():
        raise BenchmarkError("Target-free prefit cache already exists; refusing to replace")
    Z_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(Z_TRAIN_PATH, W, compressed=True)
    _save_npz_once(TRAIN_META_PATH, record_ids=ids, cycles=cycles, missing_mask=missing,
                    base_logits=z0)
    result = {"schema_version": 1, "status": "PASS_TRAIN_ONLY_TARGET_FREE_PREFIT",
              "benchmark_id": BENCHMARK_ID, "train_path": TRAIN.relative_to(ROOT).as_posix(),
              "train_sha256": TRAIN_SHA, "train_rows": len(frame), "train_cycles": [47, 79],
              "target_column_loaded": False, "target_sha256_computed": False,
              "test_or_sample_access": False, "ids_sha256": sha256_array(ids),
              "cycles_sha256": sha256_array(cycles), "missing_mask_sha256": sha256_array(missing),
              "contrast_matrix": matrix_audit, "contrast_cache_path": Z_TRAIN_PATH.relative_to(ROOT).as_posix(),
              "contrast_cache_file_sha256": sha256_file(Z_TRAIN_PATH),
              "metadata_cache_path": TRAIN_META_PATH.relative_to(ROOT).as_posix(),
              "metadata_cache_sha256": sha256_file(TRAIN_META_PATH),
              "base_logits_sha256": sha256_array(z0), "reused_state_audit": reused,
              "resource_snapshot_after_preflight": resource,
              "test_or_sample_opened": False}
    _write_once(PREFIT_AUDIT, result)
    return result


def _freeze_unsigned() -> dict[str, Any]:
    audit = _read_json(PREFIT_AUDIT)
    qa = _read_json(SYNTHETIC_QA)
    if (audit.get("status") != "PASS_TRAIN_ONLY_TARGET_FREE_PREFIT"
            or qa.get("status") != "PASS_SYNTHETIC_END_TO_END_AND_SAFE_RESUME"
            or audit.get("target_column_loaded") is not False
            or audit.get("target_sha256_computed") is not False
            or audit.get("test_or_sample_access") is not False
            or sha256_file(TRAIN) != TRAIN_SHA):
        raise BenchmarkError("Passing synthetic QA and target-free preflight are required")
    cached_z = sparse.load_npz(Z_TRAIN_PATH).tocsr().astype(np.float64)
    cached_meta = _load_npz(TRAIN_META_PATH, ("record_ids", "cycles", "missing_mask", "base_logits"))
    if (sha256_file(Z_TRAIN_PATH) != audit.get("contrast_cache_file_sha256")
            or sha256_csr(cached_z) != audit["contrast_matrix"]["contrast_sha256"]
            or sha256_file(TRAIN_META_PATH) != audit.get("metadata_cache_sha256")
            or sha256_array(cached_meta["record_ids"].astype(str)) != audit.get("ids_sha256")
            or sha256_array(cached_meta["missing_mask"].astype(bool)) != audit.get("missing_mask_sha256")
            or not np.isfinite(cached_z.data).all()):
        raise BenchmarkError("Target-free prefit feature cache does not match its compact audit")
    source_hashes = _source_hashes()
    return {"schema_version": 1, "status": "FROZEN_BEFORE_FULLTRAIN_THETA_FIT",
            "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
            "source_commit": _git("rev-parse", "HEAD"), "source_hashes": source_hashes,
            "recipe_sha256": sha256_file(RECIPE),
            "authorization_sha256": sha256_file(AUTHORIZATION),
            "synthetic_qa_sha256": sha256_file(SYNTHETIC_QA),
            "preflight_audit_sha256": sha256_file(PREFIT_AUDIT),
            "official_train_sha256": TRAIN_SHA, "official_train_rows": 50_608,
            "official_train_cycles": [47, 79], "fit_population_rows": 50_608,
            "fit_population_ids_sha256": audit["ids_sha256"],
            "expected_target_sha256_from_existing_freeze": _read_json(cgeom.TRAINED_FREEZE)["train_target_sha256"],
            "contrast_matrix_sha256": audit["contrast_matrix"]["contrast_sha256"],
            "contrast_matrix_file_sha256": audit["contrast_cache_file_sha256"],
            "metadata_cache_sha256": audit["metadata_cache_sha256"],
            "base_logits_sha256": audit["base_logits_sha256"],
            "reused_state_audit": audit["reused_state_audit"],
            "optimizer": {"name": "scipy.optimize.minimize:L-BFGS-B", "C": 1.0,
                          "objective": "sum(logaddexp(0,-(2*y-1)*(z0+Z@theta))) + theta@theta/2",
                          "initial_theta": "zeros", "intercept": False,
                          "options": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}},
            "policy": {"present_stream": ".5*expit(C2B_logit+Z@theta)+.5*original_A02",
                       "missing_stream": "exact saved full-Train C1_B HGB probability",
                       "threshold": 0.5, "rule": "probability >= 0.50"},
            "fit_count": 1, "prerequisite_refit_count": 0,
            "test_features_or_probabilities_opened": False, "test_labels_opened": False,
            "sample_submission_opened": False, "kaggle_contacted": False}


def make_prefit_freeze() -> dict[str, Any]:
    unsigned = _freeze_unsigned()
    value = {**unsigned, "freeze_sha256": _sha_obj(unsigned),
             "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(PREFIT_FREEZE, value)
    return value


def _verify_signed(path: Path, signature_key: str) -> dict[str, Any]:
    value = _read_json(path)
    signature = value.get(signature_key)
    unsigned = {key: item for key, item in value.items()
                if key not in {signature_key, "created_at_utc"}}
    if signature != _sha_obj(unsigned):
        raise BenchmarkError(f"Signed freeze verification failed: {path}")
    return value


def _old_release_audit(events: list[dict[str, Any]]) -> dict[str, Any]:
    reserves = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                and e.get("reservation_key") == OLD_RELEASE_KEY]
    releases = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"
                and e.get("reservation_key") == OLD_RELEASE_KEY]
    old_fit_events = [e for e in events if e.get("fit_id") == "POS_CONTRAST_V1_E79_REAL"]
    if (len(reserves) != 1 or len(releases) != 1
            or any(e.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"} for e in old_fit_events)):
        raise BenchmarkError("Historical conditional E79 reservation/fit state is not the reviewed released state")
    return {"reservation_key": OLD_RELEASE_KEY, "reserve_event_sha256": _sha_obj(reserves[0]),
            "release_event_sha256": _sha_obj(releases[0]), "released": True,
            "fit_started": False, "reused": False}


def reserve_fit() -> dict[str, Any]:
    _require_clean_tree()
    freeze = _verify_signed(PREFIT_FREEZE, "freeze_sha256")
    if not _committed_file(PREFIT_FREEZE):
        raise BenchmarkError("Committed pre-fit freeze is required before budget reservation")
    events = _ledger_events()
    before = _ledger_snapshot(events)
    old_release = _old_release_audit(events)
    own_keys = [e for e in events if e.get("reservation_key") == RESERVATION_KEY]
    own_ids = [e for e in events if e.get("fit_id") == FIT_ID]
    if own_ids and not (own_keys and not any(e.get("event") == "FIT_STARTED" for e in own_ids)):
        raise BenchmarkError("Unique fit ID already has an ambiguous or started ledger state; no retry")
    if before.get("fit_pending") != 0:
        raise BenchmarkError(f"Another project fit is pending; do not overlap: {before}")
    caps = _category_caps(before)
    active = before.get("active_reservations_by_category", {})
    if (before.get("active_reservations", 0) >= before.get("global_cap", 0)
            or active.get("pilot", 0) >= caps.get("pilot", 0)):
        raise BenchmarkError(f"No live shared budget slot is available: {before}")
    if (len([e for e in own_keys if e.get("event") == "CARD_FITS_RESERVED"]) > 1
            or any(e.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"} for e in own_ids)):
        raise BenchmarkError("One-fit authorization already consumed or reservation duplicated")
    if own_keys:
        reserved = next((e for e in own_keys if e.get("event") == "CARD_FITS_RESERVED"), None)
        if (reserved is None or reserved.get("fit_ids") != [FIT_ID]
                or reserved.get("freeze_signature") != freeze["freeze_sha256"]):
            raise BenchmarkError("Existing reservation does not match this frozen one-fit recipe")
        result = {"schema_version": 1, "status": "ONE_FIT_RESERVED", "benchmark_id": BENCHMARK_ID,
                  "fit_id": FIT_ID, "reservation_key": RESERVATION_KEY,
                  "reservation_event": reserved, "baseline_snapshot": before,
                  "idempotent_resume": True, "historical_released_slot_reused": False,
                  "old_release_audit": old_release}
        _write_once(RESERVATION_REPORT, result)
        return result
    event = {"event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
             "card_id": BENCHMARK_ID, "stage": "REAL_E79_FULLTRAIN_THETA", "category": "pilot",
             "reservation_key": RESERVATION_KEY, "fits": 1, "fit_ids": [FIT_ID],
             "freeze_signature": freeze["freeze_sha256"], "source_commit": freeze["source_commit"],
             "reason": "Exactly one user-authorized REAL POS_CONTRAST full-Train exploratory theta fit; prior conditional slot remains released.",
             "at_utc": datetime.now(timezone.utc).isoformat()}
    before_sha = sha256_file(LEDGER)
    # Reconcile a concurrent reservation or fit before appending. Do not revive the old key.
    current_events = _ledger_events()
    current_snapshot = _ledger_snapshot(current_events)
    current_caps = _category_caps(current_snapshot)
    if (sha256_file(LEDGER) != before_sha or current_snapshot.get("fit_pending") != 0
            or current_snapshot.get("active_reservations", 0) >= current_snapshot.get("global_cap", 0)
            or current_snapshot.get("active_reservations_by_category", {}).get("pilot", 0)
                >= current_caps.get("pilot", 0)):
        raise BenchmarkError("Shared ledger changed or capacity was consumed during reservation precheck")
    _append_event(event)
    after = _ledger_events()
    after_snapshot = _ledger_snapshot(after)
    if (after_snapshot.get("active_reservations", 0) > after_snapshot.get("global_cap", 0)
            or after_snapshot.get("active_reservations_by_category", {}).get("pilot", 0)
                > _category_caps(after_snapshot).get("pilot", 0)):
        raise BenchmarkError("Reservation exceeded a shared budget cap")
    result = {"schema_version": 1, "status": "ONE_FIT_RESERVED", "benchmark_id": BENCHMARK_ID,
              "fit_id": FIT_ID, "reservation_key": RESERVATION_KEY, "reservation_event": event,
              "baseline_ledger_sha256": before_sha, "baseline_snapshot": before,
              "post_ledger_sha256": sha256_file(LEDGER), "post_snapshot": after_snapshot,
              "idempotent_resume": False, "historical_released_slot_reused": False,
              "old_release_audit": old_release}
    _write_once(RESERVATION_REPORT, result)
    return result


def run_fulltrain_fit() -> dict[str, Any]:
    _require_clean_tree()
    freeze = _verify_signed(PREFIT_FREEZE, "freeze_sha256")
    reservation = _read_json(RESERVATION_REPORT)
    if (not _committed_file(PREFIT_FREEZE) or not _committed_file(RESERVATION_REPORT)
            or reservation.get("status") != "ONE_FIT_RESERVED"
            or reservation.get("fit_id") != FIT_ID):
        raise BenchmarkError("Committed pre-fit freeze and one-fit reservation are required")
    _assert_sources(freeze["source_hashes"])
    events = _ledger_events()
    own_res = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
               and e.get("reservation_key") == RESERVATION_KEY]
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == FIT_ID]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == FIT_ID]
    if (len(own_res) != 1 or starts or terminals or THETA_PATH.exists()
            or FIT_RECEIPT.exists() or TRAINED_FREEZE.exists()):
        raise BenchmarkError("This one-shot theta fit has already started or left output; no retry")
    if (own_res[0].get("fit_ids") != [FIT_ID]
            or own_res[0].get("freeze_signature") != freeze.get("freeze_sha256")
            or own_res[0].get("category") != "pilot"):
        raise BenchmarkError("Live one-fit reservation does not match the committed frozen recipe")
    before = _ledger_snapshot(events)
    if before.get("fit_pending") != 0:
        raise BenchmarkError(f"Another fit is currently pending; stop before training: {before}")
    theta_cache = sparse.load_npz(Z_TRAIN_PATH).tocsr().astype(np.float64)
    arrays = _load_npz(TRAIN_META_PATH, ("record_ids", "cycles", "missing_mask", "base_logits"))
    estimate = max(1024**3, 6 * (theta_cache.data.nbytes + theta_cache.indices.nbytes + theta_cache.indptr.nbytes)
                   + 512 * 1024**2)
    resource = _resource_gate(initial=True, estimated_incremental=estimate)
    if (sha256_file(Z_TRAIN_PATH) != freeze["contrast_matrix_file_sha256"]
            or sha256_csr(theta_cache) != freeze["contrast_matrix_sha256"]
            or sha256_file(TRAIN_META_PATH) != freeze["metadata_cache_sha256"]
            or sha256_array(arrays["record_ids"].astype(str)) != freeze["fit_population_ids_sha256"]
            or len(arrays["cycles"]) != 50_608):
        raise BenchmarkError("Committed target-free training cache differs from freeze")
    # Target column is first opened after the pre-fit freeze and unique reservation are committed.
    y_frame = pd.read_csv(TRAIN, usecols=["artifact_signal"])
    y = pd.to_numeric(y_frame.artifact_signal, errors="raise").to_numpy(dtype=np.int8)
    target_sha = sha256_array(y)
    if (len(y) != 50_608 or not np.isin(y, [0, 1]).all()
            or target_sha != freeze["expected_target_sha256_from_existing_freeze"]):
        raise BenchmarkError("Official Train labels do not match the previously verified target population")
    events = _ledger_events()
    live_sha = sha256_file(LEDGER)
    live_snapshot = _ledger_snapshot(events)
    if live_snapshot.get("fit_pending") != 0 or any(e.get("fit_id") == FIT_ID for e in events):
        raise BenchmarkError("Live ledger changed before the one authorized fit start")
    start_event = {"event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
                   "card_id": BENCHMARK_ID, "stage": "REAL_E79_FULLTRAIN_THETA",
                   "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID,
                   "fit_kind": "PREDICTOR", "fit_consumed_at_start": True,
                   "fit_population": {"rows": len(y), "cycles": [47, 79],
                                      "record_ids_sha256": sha256_array(arrays["record_ids"].astype(str)),
                                      "target_sha256": target_sha, "target_read_purpose": "fulltrain_contrast_theta_fit_only",
                                      "contrast_matrix_sha256": sha256_csr(theta_cache),
                                      "base_logits_sha256": sha256_array(arrays["base_logits"])},
                   "resource_snapshot": resource, "source_commit": freeze["source_commit"],
                   "freeze_signature": freeze["freeze_sha256"],
                   "pre_start_ledger_sha256": live_sha,
                   "at_utc": datetime.now(timezone.utc).isoformat()}
    if sha256_file(LEDGER) != live_sha:
        raise BenchmarkError("Shared ledger changed before fit-start append")
    _append_event(start_event)
    try:
        with threadpool_limits(limits=THREADS):
            theta, optimizer = fit_offset_coefficients(theta_cache, arrays["base_logits"], y, c=1.0)
        if not optimizer.get("success") or not np.isfinite(theta).all():
            raise BenchmarkError(f"One-shot L-BFGS-B theta failed acceptance: {optimizer}")
        _save_npz_once(THETA_PATH, theta=theta)
        receipt = {"schema_version": 1, "status": "PASS_ONE_FULLTRAIN_THETA_FIT_SAVED",
                   "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
                   "source_commit": freeze["source_commit"], "source_hashes": freeze["source_hashes"],
                   "pre_fit_freeze_sha256": freeze["freeze_sha256"],
                   "reservation_key": RESERVATION_KEY, "train_sha256": TRAIN_SHA,
                   "fit_rows": len(y), "fit_cycles": [47, 79],
                   "fit_ids_sha256": sha256_array(arrays["record_ids"].astype(str)),
                   "target_sha256": target_sha, "contrast_matrix_sha256": sha256_csr(theta_cache),
                   "base_logits_sha256": sha256_array(arrays["base_logits"]),
                   "theta_sha256": sha256_array(theta), "theta_file_sha256": sha256_file(THETA_PATH),
                   "optimizer": optimizer, "resource_before_fit": resource,
                   "official_theta_fit_count": 1, "prerequisite_refit_count": 0,
                   "test_features_or_probabilities_opened": False, "test_labels_opened": False,
                   "sample_submission_opened": False, "kaggle_contacted": False,
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_once(FIT_RECEIPT, receipt)
        unsigned = {"schema_version": 1, "status": "THETA_FIT_FROZEN_BEFORE_TEST_INFERENCE",
                    "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
                    "source_commit": freeze["source_commit"],
                    "pre_fit_freeze_sha256": sha256_file(PREFIT_FREEZE),
                    "fit_receipt_sha256": sha256_file(FIT_RECEIPT),
                    "theta_sha256": sha256_array(theta), "theta_file_sha256": sha256_file(THETA_PATH),
                    "train_sha256": TRAIN_SHA, "fit_rows": 50_608,
                    "target_sha256": target_sha, "fit_population_cycles": [47, 79],
                    "test_or_sample_rows_loaded": False, "test_labels_loaded": False,
                    "inference_started": False, "kaggle_contacted": False}
        trained = {**unsigned, "freeze_sha256": _sha_obj(unsigned),
                   "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _write_once(TRAINED_FREEZE, trained)
        _append_event({"event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
                       "card_id": BENCHMARK_ID, "stage": "REAL_E79_FULLTRAIN_THETA",
                       "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID,
                       "fit_kind": "PREDICTOR", "receipt_sha256": sha256_file(FIT_RECEIPT),
                       "source_commit": freeze["source_commit"],
                       "freeze_signature": freeze["freeze_sha256"]})
        return {"status": "PASS_ONE_FULLTRAIN_THETA_AND_TRAINED_FREEZE_SAVED",
                "fit_receipt": receipt, "trained_freeze": trained,
                "ledger_fit_saved": True}
    except Exception as exc:
        current = _ledger_events()
        has_terminal = any(e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == FIT_ID
                           for e in current)
        if not has_terminal:
            _append_event({"event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                           "card_id": BENCHMARK_ID, "stage": "REAL_E79_FULLTRAIN_THETA",
                           "reservation_key": RESERVATION_KEY, "fit_id": FIT_ID,
                           "fit_kind": "PREDICTOR", "reason": f"one-shot fit failed: {type(exc).__name__}: {exc}",
                           "source_commit": freeze["source_commit"],
                           "freeze_signature": freeze["freeze_sha256"]})
        raise


def _verify_trained_freeze() -> dict[str, Any]:
    if not _committed_file(TRAINED_FREEZE) or not _committed_file(FIT_RECEIPT):
        raise BenchmarkError("Committed trained-state freeze and fit receipt are required before Test inference")
    freeze = _verify_signed(TRAINED_FREEZE, "freeze_sha256")
    if (freeze.get("status") != "THETA_FIT_FROZEN_BEFORE_TEST_INFERENCE"
            or freeze.get("fit_id") != FIT_ID
            or sha256_file(THETA_PATH) != freeze.get("theta_file_sha256")
            or sha256_array(_load_npz(THETA_PATH, ("theta",))["theta"]) != freeze.get("theta_sha256")):
        raise BenchmarkError("Trained theta state or receipt has changed")
    receipt = _read_json(FIT_RECEIPT)
    if (receipt.get("status") != "PASS_ONE_FULLTRAIN_THETA_FIT_SAVED"
            or receipt.get("fit_id") != FIT_ID
            or receipt.get("theta_file_sha256") != sha256_file(THETA_PATH)):
        raise BenchmarkError("Saved one-shot fit receipt is invalid")
    return freeze


def _test_outputs() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    resource = _resource_gate(initial=False, estimated_incremental=1024**3)
    if sha256_file(TEST) != TEST_SHA or sha256_file(SAMPLE) != SAMPLE_SHA:
        raise BenchmarkError("Official Test or SampleSubmission hash mismatch")
    test_columns = list(dict.fromkeys(["record_id", "cycle_code", *cgeom.CLASSIFICATION_FEATURES]))
    test = pd.read_csv(TEST, usecols=test_columns, dtype={"record_id": str})
    if "artifact_signal" in test.columns or len(test) != 29_421:
        raise BenchmarkError("Test feature-only read exposed a target or invalid row count")
    ids = test.record_id.astype(str).to_numpy(dtype=str)
    cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if (test.record_id.isna().any() or (ids == "").any() or test.record_id.duplicated().any()
            or not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64))):
        raise BenchmarkError("Official Test ID/cycle support failed")
    sample_header = list(pd.read_csv(SAMPLE, nrows=0).columns)
    sample_ids = pd.read_csv(SAMPLE, usecols=["record_id"], dtype={"record_id": str}).record_id.astype(str).to_numpy()
    if (sample_header != ["record_id", "artifact_signal"] or len(sample_ids) != len(ids)
            or not np.array_equal(ids, sample_ids) or len(np.unique(sample_ids)) != len(sample_ids)):
        raise BenchmarkError("Official SampleSubmission schema or ordered Test IDs differ")
    model = joblib.load(cgeom.A02_MODEL)
    vectorizer = model.text_space.vectorizer
    Z, z0, missing, matrix_audit = _build_contrast(test, vectorizer=vectorizer)
    theta = _load_npz(THETA_PATH, ("theta",))["theta"].astype(np.float64)
    if Z.shape[1] != len(theta):
        raise BenchmarkError("Test contrast width differs from fitted theta")
    if (sha256_file(cgeom.OUTPUT_NPZ) != PREDICTION_BUNDLE_SHA
            or sha256_file(cgeom.A02_TEST_NPZ) != A02_SOURCE_TEST_NPZ_SHA):
        raise BenchmarkError("Verified A02/C1_B Test prediction bundle hashes changed")
    arrays = _load_npz(cgeom.OUTPUT_NPZ,
                       ("record_ids", "missing_mask", "a02_probability", "c1b_missing_probability"))
    source_a02 = _load_npz(cgeom.A02_TEST_NPZ, ("record_ids", "candidate_probability"))
    if (not np.array_equal(ids, arrays["record_ids"].astype(str))
            or not np.array_equal(ids, source_a02["record_ids"].astype(str))
            or not np.array_equal(arrays["a02_probability"], source_a02["candidate_probability"])
            or not np.array_equal(missing, arrays["missing_mask"].astype(bool))):
        raise BenchmarkError("Cached A02/C1_B predictions or availability mask do not align with Test rows")
    p_a02 = arrays["a02_probability"].astype(np.float64)
    p_c1b = arrays["c1b_missing_probability"].astype(np.float64)
    if (p_a02.shape != (len(ids),) or p_c1b.shape != (len(ids),)
            or not np.isfinite(p_a02).all() or not np.isfinite(p_c1b).all()
            or np.any((p_a02 < 0) | (p_a02 > 1)) or np.any((p_c1b < 0) | (p_c1b > 1))):
        raise BenchmarkError("Saved A02/C1_B probabilities are invalid")
    gamma = _load_npz(THETA_PATH, ("theta",))["theta"].astype(np.float64)
    base_probability = expit(z0)
    q = expit(z0 + Z @ gamma)
    gamma0_p = np.where(missing, p_c1b, 0.5 * base_probability + 0.5 * p_a02)
    p = np.where(missing, p_c1b, 0.5 * q + 0.5 * p_a02)
    labels = (p >= 0.50).astype(np.int8)
    if (not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.array_equal(p[missing], p_c1b[missing])
            or labels.dtype != np.int8 or not np.isin(labels, [0, 1]).all()):
        raise BenchmarkError("E79 fixed policy produced invalid probabilities or labels")
    out = {"record_ids": ids, "cycles": cycles, "missing_mask": missing,
           "base_logits": z0, "base_probability": base_probability,
           "contrast_sha256": np.asarray([sha256_csr(Z)]), "theta": gamma,
           "contrast_probability": q, "a02_probability": p_a02,
           "c1b_missing_probability": p_c1b, "gamma_zero_probability": gamma0_p,
           "candidate_probability": p, "labels": labels}
    meta = {"rows": len(ids), "ordered_test_ids_sha256": sha256_array(ids),
            "cycle_counts": pd.Series(cycles).value_counts().sort_index().to_dict(),
            "present_rows": int((~missing).sum()), "missing_rows": int(missing.sum()),
            "test_feature_matrix": matrix_audit,
            "a02_source_test_predictions_sha256": A02_SOURCE_TEST_NPZ_SHA,
            "prediction_bundle_sha256": PREDICTION_BUNDLE_SHA,
            "sample_submission_sha256": SAMPLE_SHA, "sample_submission_columns": sample_header,
            "resource_snapshot": resource,
            "test_labels_loaded": False, "kaggle_contacted": False}
    return test, out, meta


def run_inference() -> dict[str, Any]:
    trained = _verify_trained_freeze()
    test, arrays, meta = _test_outputs()
    if CSV.exists() or PREDICTION_PATH.exists():
        raise BenchmarkError("Inference output exists; do not rebuild/replace the one authorized CSV")
    _save_npz_once(PREDICTION_PATH, **arrays)
    csv = pd.DataFrame({"record_id": arrays["record_ids"],
                        "artifact_signal": arrays["labels"].astype(np.int64)})
    _write_bytes_once(CSV, csv.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    result = {"schema_version": 1, "status": "INFERENCE_BUILT_PENDING_INDEPENDENT_VERIFICATION",
              "benchmark_id": BENCHMARK_ID,
              "csv_path": CSV.relative_to(ROOT).as_posix(), "csv_sha256": sha256_file(CSV),
              "prediction_bundle_path": PREDICTION_PATH.relative_to(ROOT).as_posix(),
              "prediction_bundle_file_sha256": sha256_file(PREDICTION_PATH),
              "input_hashes": {"train": TRAIN_SHA, "test": TEST_SHA, "sample_submission": SAMPLE_SHA},
              "trained_freeze_sha256": sha256_file(TRAINED_FREEZE),
              "theta_sha256": arrays["theta"].size and sha256_array(arrays["theta"]),
              "contrast_matrix_sha256": str(arrays["contrast_sha256"][0]),
              "base_logits_sha256": sha256_array(arrays["base_logits"]),
              "gamma_zero_probability_sha256": sha256_array(arrays["gamma_zero_probability"]),
              "candidate_probability_sha256": sha256_array(arrays["candidate_probability"]),
              "candidate_labels_sha256": sha256_array(arrays["labels"]),
              "label_counts": pd.Series(arrays["labels"]).value_counts().sort_index().to_dict(),
              "gamma_zero_replay": "PASS; exact fixed recipe, no new fit",
              "missing_fallback_exact": bool(np.array_equal(
                  arrays["candidate_probability"][arrays["missing_mask"]],
                  arrays["c1b_missing_probability"][arrays["missing_mask"]])),
              "threshold": 0.5, "rows": meta["rows"],
              "columns": ["record_id", "artifact_signal"],
              "test_labels_loaded": False, "kaggle_contacted": False,
              "test_metadata": meta, "trained_freeze": trained,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(INFERENCE_BUILD, result)
    return result


def write_results() -> dict[str, Any]:
    if not QA_REPORT.is_file() or _read_json(QA_REPORT).get("status") != "PASS_INDEPENDENT_CSV_AND_POLICY_QA":
        raise BenchmarkError("Independent inference QA must pass before results are sealed")
    receipt = _read_json(FIT_RECEIPT)
    qa = _read_json(QA_REPORT)
    manifest = {"schema_version": 1, "status": "EXPLORATORY_FULLTRAIN_E79_CSV_READY_FOR_SOL_REVIEW",
                "benchmark_id": BENCHMARK_ID, "fit_id": FIT_ID,
                "recipe_sha256": sha256_file(RECIPE),
                "source_commit": _read_json(PREFIT_FREEZE)["source_commit"],
                "result_commit": _git("rev-parse", "HEAD"),
                "train_sha256": TRAIN_SHA, "test_sha256": TEST_SHA,
                "sample_submission_sha256": SAMPLE_SHA,
                "official_theta_fit_count": 1, "prerequisite_refit_count": 0,
                "fit_receipt_sha256": sha256_file(FIT_RECEIPT),
                "trained_state_freeze_sha256": sha256_file(TRAINED_FREEZE),
                "inference_build_sha256": sha256_file(INFERENCE_BUILD),
                "independent_qa_sha256": sha256_file(QA_REPORT),
                "csv_path": CSV.relative_to(ROOT).as_posix(),
                "csv_sha256": sha256_file(CSV), "csv_rows": qa["rows"],
                "csv_columns": qa["columns"], "kaggle_status": "NOT_SUBMITTED_BY_THIS_LANE",
                "test_labels_loaded": False, "kaggle_contacted": False,
                "optimizer": receipt["optimizer"], "inference": qa,
                "interpretation": "Exploratory benchmark only; the historical POS_CONTRAST pilot gate failure remains unchanged.",
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_once(MANIFEST, manifest)
    lines = [
        "# POS_CONTRAST REAL E79 — exploratory benchmark",
        "",
        "**Status:** one full-Train theta fit completed; local Test-feature CSV passed independent structural and policy QA. This candidate remains exploratory. The historical `POS_CONTRAST_V1` pilot verdict remains `PILOT_GATE_FAILED`.",
        "",
        "## Frozen recipe",
        "",
        "The one new estimator is the REAL three-position contrast vector `theta`, fit on all 50,608 official Classification Train rows. Its fixed offset is the original C2_B intercept plus the original C2_B bag matrix multiplied by the saved coefficients. The contrast is `Z1=(first-last)/sqrt(2)` and `Z2=(first-2*middle+last)/sqrt(6)`, appended to the unchanged base and left unnormalized after projection. The optimizer is the original sum-logistic-loss plus L2/2 L-BFGS-B recipe, initialized at zero, with `maxiter=2000`, `ftol=1e-12`, `gtol=1e-6`, and `maxls=50`.",
        "",
        "Present-stream probability is `0.5 * sigmoid(C2_B logit + Z @ theta) + 0.5 * original A02 probability`. Missing-stream probability is the exact saved full-Train C1_B HGB probability. Labels use `probability >= 0.50`.",
        "",
        "## Evidence",
        "",
        f"- Train rows/cycles: {receipt['fit_rows']} / 47–79; Train SHA-256 `{TRAIN_SHA}`.",
        f"- New official fits: {receipt['official_theta_fit_count']}; prerequisite refits: {receipt['prerequisite_refit_count']}; optimizer: `{receipt['optimizer']['message']}` after {receipt['optimizer']['nit']} iterations.",
        f"- Test CSV: `{CSV.relative_to(ROOT).as_posix()}`; SHA-256 `{sha256_file(CSV)}`; {qa['rows']:,} rows in official SampleSubmission order.",
        f"- Independent QA: `{qa['status']}`; gamma-zero replay `{qa['gamma_zero_replay']}`; exact missing fallback `{qa['missing_fallback_exact']}`.",
        "- This lane did not evaluate Test labels or upload to Kaggle. No predictive metric or leaderboard result is claimed here.",
        "",
        "## Limits",
        "",
        "The frozen historical pilot failed its temporal gates. This full-Train fit is an exploratory benchmark on a different population and fixed E79 route; it does not retroactively pass those gates, qualify as a champion, or provide a Test metric. Kaggle upload remains the separate owner-controlled step.",
        "",
    ]
    text = "\n".join(lines)
    _write_bytes_once(RESULTS_MD, text.encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("synthetic", "preflight", "freeze", "reserve", "fit", "infer", "verify", "report"))
    args = parser.parse_args(argv)
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
    elif args.action == "verify":
        from scripts.verify_optimization_v3_positional_contrast_real_e79_exploratory_v1 import verify_outputs
        result = verify_outputs()
    else:
        result = write_results()
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
