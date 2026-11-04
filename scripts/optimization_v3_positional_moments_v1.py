"""Frozen four-fit POS_MOMENTS_V1 Classification pilot runner.

The script reuses only frozen parent C2_B matrices, tokenizers and BAG_LR
artifacts. New task fits learn the moment coefficients only. No Test or
SampleSubmission inputs are opened.
"""
from __future__ import annotations

import argparse
import csv
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

from scripts import optimization_v3_positional_bag_offset_v1 as offset
from scripts import optimization_v3_positional_contrast_v1 as contrast
from src.models.positional_bag_offset_v1 import fit_offset_coefficients, offset_probability
from src.models.positional_moments_v1 import (
    build_moment_matrix, canonical_sha256, direct_logit, fixed_blends,
    sha256_array, sha256_csr, sha256_file, tokenizer_idf_signature,
)


REPORT_ROOT = ROOT / "reports/optimization_v3/positional_moments_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
TRAIN = ROOT / "data/raw/classification/Train.csv"
PILOT_RUN = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
PILOT_REPORT = ROOT / "reports/optimization_v3/positional_text_pilot_v1"
BAG_RUN = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
A02_RUN = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
RECIPE = REPORT_ROOT / "RECIPE.json"
PROTOCOL = REPORT_ROOT / "SOL_EXECUTION_HANDOFF.md"
PREFLIGHT_REPORT = REPORT_ROOT / "PREFLIGHT.json"
FREEZE_REPORT = REPORT_ROOT / "FREEZE.json"
RESERVATION_REPORT = REPORT_ROOT / "RESERVATION.json"
SYNTHETIC_REPORT = REPORT_ROOT / "SYNTHETIC_QA.json"

CARD_ID = "POS_MOMENTS_V1"
AMENDMENT_ID = "POS_MOMENTS_V1_CAP142_2026_11_04"
STAGE = CARD_ID
RESERVATION_KEY = f"{CARD_ID}:PILOT_4"
FIT_IDS = (
    f"{CARD_ID}_E67_REAL", f"{CARD_ID}_E67_SHUFFLED",
    f"{CARD_ID}_E75_REAL", f"{CARD_ID}_E75_SHUFFLED",
)
ORIGINS = (67, 75)
ARMS = ("REAL", "SHUFFLED")
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_PARENT_FREEZE = "f310d0340ea4402677ca9e86aa29c43b1536ded902c0f6a6f52cb53bd333c292"
EXPECTED_LEDGER_SHA = "7816f8c64f760347cd4a5c7596a97ef99c0059a352994be00f45bce4297a986c"
BASE_CAPS = {"continuation": 50, "control_replay": 2, "pilot": 84, "technical_retry": 1}
NEW_CAPS = {"continuation": 51, "control_replay": 2, "pilot": 88, "technical_retry": 1}
BASE_SNAPSHOT = {
    "event_rows": 346, "global_cap": 137, "effective_category_caps": BASE_CAPS,
    "active_reservations_by_category": {"continuation": 49, "control_replay": 2, "pilot": 81, "technical_retry": 1},
    "active_reservations": 133, "fit_starts": 133, "fit_saved": 127, "fit_failed": 6, "fit_pending": 0,
}
NEW_PRE_FIT_SNAPSHOT = {
    "event_rows": 348, "global_cap": 142, "effective_category_caps": NEW_CAPS,
    "active_reservations_by_category": {"continuation": 49, "control_replay": 2, "pilot": 85, "technical_retry": 1},
    "active_reservations": 137, "fit_starts": 133, "fit_saved": 127, "fit_failed": 6, "fit_pending": 0,
}
AMENDED_PRE_RESERVATION_SNAPSHOT = {
    "event_rows": 347, "global_cap": 142, "effective_category_caps": NEW_CAPS,
    "active_reservations_by_category": dict(BASE_SNAPSHOT["active_reservations_by_category"]),
    "active_reservations": 133, "fit_starts": 133, "fit_saved": 127, "fit_failed": 6, "fit_pending": 0,
}
MAX_RSS = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
RUNNING_FREE_MIN = 2 * 1024**3
THREADS = 2
GATE_TOLERANCE = 1e-12


class MomentsPilotError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MomentsPilotError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(_canonical_bytes(value) + b"\n")
    with tmp.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    sparse.save_npz(tmp, sparse.csr_matrix(matrix, dtype=np.float64), compressed=True)
    with tmp.open("rb+") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_paths() -> dict[str, Path]:
    return {
        "runner": ROOT / "scripts/optimization_v3_positional_moments_v1.py",
        "verifier": ROOT / "scripts/verify_optimization_v3_positional_moments_v1.py",
        "moments_model": ROOT / "src/models/positional_moments_v1.py",
        "offset_model": ROOT / "src/models/positional_bag_offset_v1.py",
        "pilot_builder": ROOT / "src/models/optimization_v3_positional_text_pilot_v1.py",
        "classification_models": ROOT / "src/models/classification_models.py",
        "position_shuffle": ROOT / "src/models/positional_text_features.py",
        "offset_runner": ROOT / "scripts/optimization_v3_positional_bag_offset_v1.py",
        "contrast_runner": ROOT / "scripts/optimization_v3_positional_contrast_v1.py",
        "tests": ROOT / "tests/test_optimization_v3_positional_moments_v1.py",
        "recipe": RECIPE,
        "protocol": PROTOCOL,
    }


def _source_hashes() -> dict[str, str]:
    missing = [name for name, path in _source_paths().items() if not path.is_file()]
    if missing:
        raise MomentsPilotError(f"Frozen source files are missing: {missing}")
    return {name: sha256_file(path) for name, path in _source_paths().items()}


def _atomic_load_npz(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(keys) - set(archive.files))
        if missing:
            raise MomentsPilotError(f"Safe NPZ fields missing at {path}: {missing}")
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _identity_features() -> pd.DataFrame:
    columns = ["record_id", "cycle_code", "signal_stream", "text_load", "entry_friction"]
    frame = pd.read_csv(TRAIN, usecols=columns, dtype={"record_id": "string"})
    if len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise MomentsPilotError("Official Train identity/features fail count or uniqueness checks")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    if not np.isfinite(pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=np.float64)).all():
        raise MomentsPilotError("Frozen text_load subgroup requires finite official Train values")
    if not np.isin(pd.to_numeric(frame["entry_friction"], errors="raise").to_numpy(dtype=np.float64), (0, 1)).all():
        raise MomentsPilotError("Frozen entry_friction groups require binary values")
    return frame


def _load_accounting() -> dict[str, Any]:
    value = _read_json(PILOT_REPORT / "FIT_ACCOUNTING.json")
    if value.get("pilot_id") != "POSITIONAL_TEXT_PILOT_V1" or value.get("train_sha256") != EXPECTED_TRAIN_SHA:
        raise MomentsPilotError("Frozen C2_B accounting source identity changed")
    return value


def _load_parent_origin(origin: int, frame: pd.DataFrame) -> dict[str, Any]:
    if origin not in ORIGINS:
        raise MomentsPilotError(f"Unexpected origin: {origin}")
    parent_freeze = _read_json(PILOT_RUN / "FREEZE.json")
    if (parent_freeze.get("freeze_signature") != EXPECTED_PARENT_FREEZE
            or parent_freeze.get("input_sha256") != EXPECTED_TRAIN_SHA
            or parent_freeze.get("score_labels_opened") is not False
            or parent_freeze.get("test_or_sample_submission_opened") is not False):
        raise MomentsPilotError("Signed POSITIONAL_TEXT_PILOT_V1 parent boundary mismatch")
    manifest_path = PILOT_RUN / f"MATRIX_MANIFEST_E{origin}.json"
    manifest = _read_json(manifest_path)
    if (manifest.get("freeze_signature") != EXPECTED_PARENT_FREEZE
            or manifest.get("input_sha256") != EXPECTED_TRAIN_SHA):
        raise MomentsPilotError(f"E{origin} parent matrix manifest source mismatch")
    pred_path = PILOT_RUN / f"E{origin}/predictions_E{origin}.npz"
    pred_keys = ["origin", "fit_rows", "score_rows", "score_record_ids", "score_cycles",
                 "present_mask", "matrix_manifest_sha256", "prob_bag", "prob_missing_hgb"]
    pred = _atomic_load_npz(pred_path, pred_keys)
    if (int(pred["origin"].reshape(-1)[0]) != origin
            or sha256_file(manifest_path) != str(pred["matrix_manifest_sha256"].reshape(-1)[0])):
        raise MomentsPilotError(f"E{origin} parent prediction/manifest binding mismatch")
    fit_rows = np.asarray(pred["fit_rows"], dtype=np.int64)
    score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
    if (len(fit_rows) != (29705 if origin == 67 else 42117)
            or len(score_rows) != (20903 if origin == 67 else 8491)
            or len(np.intersect1d(fit_rows, score_rows))):
        raise MomentsPilotError(f"E{origin} parent row populations differ from frozen counts")
    cycles_all = frame["cycle_code"].to_numpy(dtype=np.int64)
    if (not np.array_equal(fit_rows, np.flatnonzero(cycles_all <= origin))
            or not np.array_equal(score_rows, np.flatnonzero(cycles_all > origin))):
        raise MomentsPilotError(f"E{origin} parent rows are not the exact temporal split")
    fit_ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
    score_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy()
    score_cycles = cycles_all[score_rows]
    if (not np.array_equal(score_ids, pred["score_record_ids"].astype(str))
            or not np.array_equal(score_cycles, pred["score_cycles"].astype(np.int64))):
        raise MomentsPilotError(f"E{origin} parent score identity/cycle join mismatch")
    fit_meta = manifest["fit_row_metadata"]
    score_meta = manifest["score_row_metadata"]
    for metadata, rows, ids, cycles in ((fit_meta, fit_rows, fit_ids, cycles_all[fit_rows]),
                                        (score_meta, score_rows, score_ids, score_cycles)):
        if (metadata.get("rows") != len(rows)
                or metadata.get("row_indices_sha256") != sha256_array(rows)
                or metadata.get("record_ids_sha256") != sha256_array(ids)
                or metadata.get("cycles_sha256") != sha256_array(cycles)):
            raise MomentsPilotError(f"E{origin} signed row metadata mismatch")

    matrices: dict[str, sparse.csr_matrix] = {}
    for name in ("fit_base", "score_base"):
        item = manifest["matrices"][name]
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise MomentsPilotError(f"E{origin} cached {name} file hash mismatch")
        matrix = sparse.load_npz(path).tocsr().astype(np.float64)
        if (list(matrix.shape) != item["shape"] or sha256_csr(matrix) != item["logical_sha256"]
                or not np.isfinite(matrix.data).all()):
            raise MomentsPilotError(f"E{origin} cached {name} logical integrity mismatch")
        matrices[name] = matrix

    accounting = _load_accounting()
    by_id = {item["fit_id"]: item for item in accounting["fit_units"]}
    bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
    vectorizer_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF"
    bag_acc = by_id.get(bag_id)
    vectorizer_acc = by_id.get(vectorizer_id)
    if bag_acc is None or vectorizer_acc is None:
        raise MomentsPilotError(f"E{origin} fitted BAG_LR/TF-IDF accounting missing")
    bag_artifact = PILOT_RUN / "units" / f"{bag_id}.joblib"
    bag_receipt = PILOT_RUN / "receipts" / f"{bag_id}.json"
    vectorizer_artifact = PILOT_RUN / "units" / f"{vectorizer_id}.joblib"
    vectorizer_receipt = PILOT_RUN / "receipts" / f"{vectorizer_id}.json"
    for path, acc in ((bag_artifact, bag_acc), (bag_receipt, bag_acc),
                      (vectorizer_artifact, vectorizer_acc), (vectorizer_receipt, vectorizer_acc)):
        expected = acc["artifact_sha256"] if path.suffix == ".joblib" else acc["receipt_sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise MomentsPilotError(f"E{origin} parent model/state receipt hash mismatch: {path.name}")
    bag_state = joblib.load(bag_artifact)
    bag = bag_state.get("estimator") if isinstance(bag_state, dict) else bag_state
    vectorizer_state = joblib.load(vectorizer_artifact)
    vectorizer = vectorizer_state.get("vectorizer") if isinstance(vectorizer_state, dict) else vectorizer_state
    if not hasattr(bag, "coef_") or not hasattr(vectorizer, "vocabulary_"):
        raise MomentsPilotError(f"E{origin} frozen BAG_LR/vectorizer state has an unexpected schema")
    if (list(np.asarray(bag.classes_).tolist()) != [0, 1]
            or np.asarray(bag.coef_).shape != (1, matrices["fit_base"].shape[1])
            or matrices["score_base"].shape[1] != matrices["fit_base"].shape[1]):
        raise MomentsPilotError(f"E{origin} BAG_LR class or feature shape mismatch")
    vector_sig = tokenizer_idf_signature(vectorizer)
    if vector_sig["vocabulary_idf_sha256"] != manifest["shared_preprocessing_sha256"]["tfidf_vocabulary_idf"]:
        raise MomentsPilotError(f"E{origin} fitted TF-IDF vocabulary/IDF differs from parent matrix state")
    if (bag_acc["fit_metadata"].get("coef_sha256") != sha256_array(bag.coef_)
            or bag_acc["fit_metadata"].get("intercept_sha256") != sha256_array(bag.intercept_)):
        raise MomentsPilotError(f"E{origin} BAG_LR coefficients differ from signed fit receipt")

    bag_prediction_info, bag_pred = contrast._load_bag_prediction(origin)
    a02_info, a02 = contrast._load_a02(origin, include_groups=True)
    if (not np.array_equal(a02["score_rows"], score_rows)
            or not np.array_equal(a02["score_record_ids"].astype(str), score_ids)
            or not np.array_equal(a02["score_cycles"].astype(np.int64), score_cycles)
            or not np.array_equal(a02["missing_mask"].astype(bool), ~pred["present_mask"].astype(bool))):
        raise MomentsPilotError(f"E{origin} A02 and parent score row/availability join mismatch")
    direct_score = direct_logit(matrices["score_base"], bag.coef_[0], bag.intercept_)
    if not np.allclose(expit(direct_score), pred["prob_bag"], rtol=0.0, atol=1e-12):
        raise MomentsPilotError(f"E{origin} direct float64 BAG_LR logit differs from saved C2_B prediction")
    p_old = np.asarray(bag_pred["prob_offset_real"], dtype=np.float64)
    if not np.array_equal(p_old[~pred["present_mask"].astype(bool)],
                          bag_pred["prob_missing_hgb"][~pred["present_mask"].astype(bool)]):
        raise MomentsPilotError(f"E{origin} saved missing fallback does not equal pP")

    fit_text = pd.to_numeric(frame.iloc[fit_rows]["text_load"], errors="raise").to_numpy(dtype=np.float64)
    score_text = pd.to_numeric(frame.iloc[score_rows]["text_load"], errors="raise").to_numpy(dtype=np.float64)
    edges = np.quantile(fit_text, [0.25, 0.50, 0.75], method="linear")
    groups = np.searchsorted(edges, score_text, side="right").astype(np.int8)
    if not np.array_equal(groups, a02["text_load_group"].astype(np.int8)):
        raise MomentsPilotError(f"E{origin} stored A02 subgroup cut is not reproduced from fit-prefix text_load")
    entry = pd.to_numeric(frame.iloc[score_rows]["entry_friction"], errors="raise").to_numpy(dtype=np.int8)
    if not np.array_equal(entry, a02["entry_friction"].astype(np.int8)):
        raise MomentsPilotError(f"E{origin} entry_friction subgroup differs from official Train")

    bag_receipt_obj = _read_json(bag_receipt)
    vec_receipt_obj = _read_json(vectorizer_receipt)
    if (bag_receipt_obj.get("fit_id") != bag_id or bag_receipt_obj.get("artifact_sha256") != sha256_file(bag_artifact)
            or vec_receipt_obj.get("fit_id") != vectorizer_id
            or vec_receipt_obj.get("artifact_sha256") != sha256_file(vectorizer_artifact)):
        raise MomentsPilotError(f"E{origin} parent signed fit receipt binding mismatch")
    target_sha = bag_acc["fit_population"]["fit_target_sha256"]
    return {
        "origin": origin, "manifest": manifest, "manifest_sha256": sha256_file(manifest_path),
        "prediction": pred, "prediction_bundle_sha256": sha256_file(pred_path),
        "fit_rows": fit_rows, "score_rows": score_rows, "fit_ids": fit_ids, "score_ids": score_ids,
        "score_cycles": score_cycles, "present": pred["present_mask"].astype(bool),
        "fit_base": matrices["fit_base"], "score_base": matrices["score_base"],
        "bag": bag, "bag_artifact_sha256": sha256_file(bag_artifact),
        "bag_receipt_sha256": sha256_file(bag_receipt), "bag_target_sha256": target_sha,
        "vectorizer": vectorizer, "vectorizer_artifact_sha256": sha256_file(vectorizer_artifact),
        "vectorizer_receipt_sha256": sha256_file(vectorizer_receipt), "tokenizer_signature": vector_sig,
        "bag_prediction_info": bag_prediction_info, "bag_prediction": bag_pred,
        "a02_info": a02_info, "a02": a02,
        "p_old": p_old, "p_a02": np.asarray(a02["prediction"], dtype=np.float64),
        "entry_friction": entry, "text_load_group": groups, "text_load_edges": edges.tolist(),
        "z0_fit": direct_logit(matrices["fit_base"], bag.coef_[0], bag.intercept_),
        "z0_score": direct_score,
    }


def _cache_path(origin: int, name: str) -> Path:
    return RUN_ROOT / f"E{origin}" / "matrices" / f"{name}.npz"


def preflight() -> dict[str, Any]:
    source_hashes = _source_hashes()
    if sha256_file(TRAIN) != EXPECTED_TRAIN_SHA:
        raise MomentsPilotError("Official Classification Train hash differs from frozen input")
    frame = _identity_features()
    start = time.perf_counter()
    before = _resource_snapshot(0)
    origins: dict[str, Any] = {}
    for origin in ORIGINS:
        state = _load_parent_origin(origin, frame)
        docs_fit = state["fit_rows"]
        docs_score = state["score_rows"]
        from src.models.classification_models import _text_values
        all_docs, missing = _text_values(frame["signal_stream"])
        if len(all_docs) != len(frame) or not np.array_equal(missing[docs_score], ~state["present"]):
            raise MomentsPilotError(f"E{origin} frozen stream availability does not match normalized text")
        documents = [all_docs[int(i)] for i in np.concatenate((docs_fit, docs_score))]
        split = len(docs_fit)
        arm_audits: dict[str, Any] = {}
        matrices_for_origin: dict[str, sparse.csr_matrix] = {}
        for arm in ARMS:
            whole, audit = build_moment_matrix(documents, state["vectorizer"], shuffled=(arm == "SHUFFLED"), seed=42)
            fit_matrix = whole[:split].tocsr()
            score_matrix = whole[split:].tocsr()
            del whole
            for label, matrix in (("fit", fit_matrix), ("score", score_matrix)):
                if matrix.shape[1] != 3 * state["tokenizer_signature"]["unigram_size"]:
                    raise MomentsPilotError(f"E{origin} {arm} {label} moments width is not 3*d")
                if not np.isfinite(matrix.data).all():
                    raise MomentsPilotError(f"E{origin} {arm} {label} moments are non-finite")
                _atomic_sparse(_cache_path(origin, f"{label}_{arm.lower()}"), matrix)
                matrices_for_origin[f"{label}_{arm.lower()}"] = matrix
            arm_audits[arm.lower()] = {
                **audit,
                "fit_resource_estimate": _fit_memory_estimate(fit_matrix),
                "fit": {"shape": list(fit_matrix.shape), "sha256": sha256_csr(fit_matrix),
                        "file_sha256": sha256_file(_cache_path(origin, f"fit_{arm.lower()}"))},
                "score": {"shape": list(score_matrix.shape), "sha256": sha256_csr(score_matrix),
                          "file_sha256": sha256_file(_cache_path(origin, f"score_{arm.lower()}"))},
            }
        if (arm_audits["real"]["bag_channel_sha256"] != arm_audits["shuffled"]["bag_channel_sha256"]
                or arm_audits["real"]["token_count_sha256"] != arm_audits["shuffled"]["token_count_sha256"]
                or arm_audits["real"]["documents_sha256"] != arm_audits["shuffled"]["documents_sha256"]):
            raise MomentsPilotError(f"E{origin} shuffle changed frozen bag, document identity, or token length")
        offset_data = {
            "z0_fit": state["z0_fit"], "z0_score": state["z0_score"],
        }
        for name, value in offset_data.items():
            path = RUN_ROOT / f"E{origin}" / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".npy.tmp")
            with tmp.open("wb") as stream:
                np.save(stream, np.asarray(value, dtype=np.float64), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        origins[f"E{origin}"] = {
            "fit_rows": len(state["fit_rows"]), "score_rows": len(state["score_rows"]),
            "fit_rows_sha256": sha256_array(state["fit_rows"]), "score_rows_sha256": sha256_array(state["score_rows"]),
            "fit_ids_sha256": sha256_array(state["fit_ids"]), "score_ids_sha256": sha256_array(state["score_ids"]),
            "score_cycles_sha256": sha256_array(state["score_cycles"]),
            "manifest_sha256": state["manifest_sha256"],
            "prediction_bundle_sha256": state["prediction_bundle_sha256"],
            "fit_base_sha256": sha256_csr(state["fit_base"]), "score_base_sha256": sha256_csr(state["score_base"]),
            "bag_artifact_sha256": state["bag_artifact_sha256"], "bag_receipt_sha256": state["bag_receipt_sha256"],
            "fit_target_sha256": state["bag_target_sha256"],
            "vectorizer_artifact_sha256": state["vectorizer_artifact_sha256"],
            "vectorizer_receipt_sha256": state["vectorizer_receipt_sha256"],
            "tokenizer_signature": state["tokenizer_signature"],
            "bag_prediction_ready_sha256": state["bag_prediction_info"]["ready_sha256"],
            "bag_prediction_bundle_sha256": state["bag_prediction_info"]["output_sha256"],
            "a02_metadata_sha256": state["a02_info"]["prediction_json_sha256"],
            "a02_receipt_sha256": state["a02_info"]["receipt_sha256"],
            "a02_probability_sha256": state["a02_info"]["safe_array_hashes"]["prediction"],
            "missing_rows": int((~state["present"]).sum()), "present_rows": int(state["present"].sum()),
            "text_load_fit_prefix_edges": state["text_load_edges"],
            "moments": arm_audits,
            "z0_fit_sha256": sha256_array(state["z0_fit"]), "z0_score_sha256": sha256_array(state["z0_score"]),
        }
        del matrices_for_origin, state, documents, all_docs
        gc.collect()
    after = _resource_snapshot(0)
    estimate_rows = [origins[f"E{o}"]["moments"][arm.lower()]["fit_resource_estimate"]["estimated_incremental_bytes"]
                     for o in ORIGINS for arm in ARMS]
    fit_resource_estimates_pass = all(
        before["projected_rss_bytes"] + estimate <= MAX_RSS
        and before["available_ram_bytes"] - estimate >= RUNNING_FREE_MIN
        for estimate in estimate_rows
    )
    result = {
        "schema_version": 1, "card_id": CARD_ID, "status": "PASS_TARGET_FREE_PREFLIGHT",
        "source_commit": _git("rev-parse", "HEAD"), "source_hashes": source_hashes,
        "input_sha256": EXPECTED_TRAIN_SHA, "input_rows": 50608,
        "source_boundary": "Official Classification Train features only; artifact_signal not selected or read; no Test/Sample access",
        "parent_freeze_signature": EXPECTED_PARENT_FREEZE,
        "origins": origins,
        "runtime_seconds": float(time.perf_counter() - start),
        "resource_before": before, "resource_after": after,
        "fit_resource_estimates_pass": fit_resource_estimates_pass,
        "fit_resource_estimate_basis": "max(1 GiB, 4 * CSR matrix bytes + 256 MiB) per fit; dynamic guard repeats immediately before each fit",
        "prelabel_predictions_created": False, "query_labels_read": False,
        "test_or_sample_access": False,
        "interpretation_limit": "Target-free shape, lineage, tokenization, bag/length preservation and resource diagnostics; not an outcome gate.",
    }
    _atomic_json(PREFLIGHT_REPORT, result)
    _atomic_json(RUN_ROOT / "PREFLIGHT.json", result)
    return result


def _ledger_snapshot(path: Path = LEDGER) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = offset._ledger_events(path)
    return events, offset.ledger_snapshot(events)


def _assert_baseline_ledger(path: Path = LEDGER, *, freeze_sha: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise MomentsPilotError("Shared Research Next ledger is missing")
    digest = sha256_file(path)
    if freeze_sha is None and digest != EXPECTED_LEDGER_SHA:
        raise MomentsPilotError(f"Shared ledger differs from authorized baseline: {digest}")
    events, snapshot = _ledger_snapshot(path)
    if snapshot != BASE_SNAPSHOT:
        raise MomentsPilotError(f"Shared ledger snapshot differs from protocol: {snapshot}")
    if any(e.get("card_id") == CARD_ID or e.get("reservation_key") == RESERVATION_KEY for e in events):
        raise MomentsPilotError("POS_MOMENTS_V1 has prior ledger events; no duplicate reservation or retry")
    if freeze_sha is not None and digest != freeze_sha:
        raise MomentsPilotError(f"Ledger changed after source freeze: expected {freeze_sha}, found {digest}")
    return events, snapshot


def _resource_snapshot(estimate_bytes: int) -> dict[str, Any]:
    proc = psutil.Process()
    rss = int(proc.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    return {
        "pid": proc.pid, "process_rss_bytes": rss, "available_ram_bytes": available,
        "estimated_incremental_memory_bytes": int(estimate_bytes),
        "projected_rss_bytes": rss + int(estimate_bytes),
        "projected_available_ram_bytes": available - int(estimate_bytes),
        "jobs": 1, "threads": THREADS, "rss_ceiling_bytes": MAX_RSS,
        "initial_available_ram_min_bytes": INITIAL_FREE_MIN, "running_reserve_bytes": RUNNING_FREE_MIN,
    }


def _fit_memory_estimate(matrix: sparse.spmatrix) -> dict[str, int]:
    x = sparse.csr_matrix(matrix, dtype=np.float64, copy=False)
    matrix_bytes = int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)
    estimated = max(1024**3, matrix_bytes * 4 + 256 * 1024**2)
    return {"matrix_bytes": matrix_bytes, "estimated_incremental_bytes": estimated}


def _check_fit_resources(*, estimate_bytes: int, initial: bool) -> dict[str, Any]:
    snap = _resource_snapshot(estimate_bytes)
    if initial and snap["available_ram_bytes"] < INITIAL_FREE_MIN:
        raise MomentsPilotError(f"RESOURCE_BLOCK_BEFORE_FIT: initial free RAM {snap['available_ram_bytes']} < 4 GiB")
    if snap["projected_rss_bytes"] > MAX_RSS or snap["projected_available_ram_bytes"] < RUNNING_FREE_MIN:
        raise MomentsPilotError(
            "RESOURCE_BLOCK_BEFORE_FIT: projected RSS/free-RAM guard failed "
            f"(rss={snap['projected_rss_bytes']}, free={snap['projected_available_ram_bytes']}, min_free={RUNNING_FREE_MIN})"
        )
    return snap


def _assert_code_commit_ancestor(source_commit: str) -> None:
    completed = subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                               cwd=ROOT, capture_output=True)
    if completed.returncode != 0:
        raise MomentsPilotError("Frozen source commit is not an ancestor of current HEAD")


def freeze() -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise MomentsPilotError("Commit recipe/code/tests and target-free preflight before freezing")
    preflight = _read_json(PREFLIGHT_REPORT)
    if preflight.get("status") != "PASS_TARGET_FREE_PREFLIGHT" or preflight.get("query_labels_read") is not False:
        raise MomentsPilotError("Target-free preflight is not PASS/label-closed")
    if preflight.get("fit_resource_estimates_pass") is not True:
        raise MomentsPilotError("Target-free per-fit memory estimates do not meet frozen RAM/RSS limits")
    source_hashes = _source_hashes()
    if source_hashes != preflight.get("source_hashes"):
        raise MomentsPilotError("Code/recipe/test files changed after target-free preflight")
    synthetic = _read_json(SYNTHETIC_REPORT)
    if (synthetic.get("status") != "PASS" or synthetic.get("production_ledger_unchanged") is not True
            or synthetic.get("source_hashes") != source_hashes
            or synthetic.get("source_commit") != preflight.get("source_commit")):
        raise MomentsPilotError("Actual-entrypoint synthetic QA must PASS before source freeze")
    events, snapshot = _assert_baseline_ledger()
    if sha256_file(TRAIN) != EXPECTED_TRAIN_SHA:
        raise MomentsPilotError("Official Train changed before source freeze")
    prefix_copy = RUN_ROOT / "ledger_pre_reservation.jsonl"
    prefix_copy.parent.mkdir(parents=True, exist_ok=True)
    prefix_tmp = prefix_copy.with_suffix(".jsonl.tmp")
    shutil.copyfile(LEDGER, prefix_tmp)
    if sha256_file(prefix_tmp) != EXPECTED_LEDGER_SHA:
        raise MomentsPilotError("Frozen pre-reservation ledger copy hash mismatch")
    os.replace(prefix_tmp, prefix_copy)
    source_commit = str(preflight.get("source_commit", ""))
    if not source_commit:
        raise MomentsPilotError("Preflight lacks the code source commit")
    _assert_code_commit_ancestor(source_commit)
    synthetic_source = str(synthetic.get("source_commit", ""))
    if not synthetic_source:
        raise MomentsPilotError("Synthetic QA lacks its executing source commit")
    _assert_code_commit_ancestor(synthetic_source)
    if _source_hashes() != preflight.get("source_hashes"):
        raise MomentsPilotError("Synthetic/preflight code identities differ")
    cache_paths = [list((RUN_ROOT / f"E{o}/matrices").glob("*.npz")) for o in ORIGINS]
    cache_hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path)
                    for group in cache_paths for path in group}
    expected_cache_hashes = {
        f"experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1/E{o}/matrices/{name}_{arm.lower()}.npz":
        preflight["origins"][f"E{o}"]["moments"][arm.lower()][name]["file_sha256"]
        for o in ORIGINS for arm in ARMS for name in ("fit", "score")
    }
    if cache_hashes != expected_cache_hashes:
        raise MomentsPilotError("Preflight matrix cache file set/hash differs from target-free report")
    body = {
        "schema_version": 1, "card_id": CARD_ID, "freeze_type": "PREFIT_SOURCE_AND_INPUT_FREEZE",
        "source_commit": source_commit, "freeze_created_from_commit": _git("rev-parse", "HEAD"),
        "source_hashes": source_hashes, "input_sha256": EXPECTED_TRAIN_SHA, "input_rows": 50608,
        "parent_freeze_signature": EXPECTED_PARENT_FREEZE,
        "preflight_sha256": sha256_file(PREFLIGHT_REPORT), "synthetic_qa_sha256": sha256_file(SYNTHETIC_REPORT),
        "pre_reservation_ledger_sha256": sha256_file(LEDGER), "pre_reservation_ledger_bytes": LEDGER.stat().st_size,
        "pre_reservation_ledger_rows": len(events), "pre_reservation_ledger_snapshot": snapshot,
        "pre_reservation_ledger_copy_sha256": sha256_file(prefix_copy),
        "effective_caps_before": BASE_CAPS, "effective_caps_after_authorized_amendment": NEW_CAPS,
        "global_cap_before": 137, "global_cap_after_authorized_amendment": 142,
        "pilot_fit_ids": list(FIT_IDS), "pilot_fit_count": 4,
        "conditional_e79_fit_id": f"{CARD_ID}_E79_REAL",
        "conditional_e79_reserved_now": False, "conditional_e79_start_authorized": False,
        "matrix_cache_hashes": cache_hashes, "origins": preflight["origins"],
        "query_labels_read": False, "test_or_sample_access": False,
        "fit_order": list(FIT_IDS),
        "fixed_policy": "present=.5*expit(bag_logit+W@gamma)+.5*A02; missing=original pP exactly; threshold >=.50",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["freeze_signature"] = _sha_obj(body)
    _atomic_json(FREEZE_REPORT, body)
    _atomic_json(RUN_ROOT / "FREEZE.json", body)
    return body


def _load_freeze() -> dict[str, Any]:
    freeze_data = _read_json(FREEZE_REPORT)
    signature = freeze_data.get("freeze_signature")
    unsigned = dict(freeze_data)
    unsigned.pop("freeze_signature", None)
    if signature != _sha_obj(unsigned):
        raise MomentsPilotError("POS_MOMENTS_V1 freeze signature mismatch")
    if freeze_data.get("source_hashes") != _source_hashes():
        raise MomentsPilotError("Frozen POS_MOMENTS_V1 source hash changed")
    if sha256_file(TRAIN) != freeze_data.get("input_sha256"):
        raise MomentsPilotError("Official Train input differs from freeze")
    _assert_code_commit_ancestor(str(freeze_data["source_commit"]))
    run_freeze = _read_json(RUN_ROOT / "FREEZE.json")
    if run_freeze != freeze_data:
        raise MomentsPilotError("Tracked/run freeze copies differ")
    return freeze_data


def _append_event(path: Path, event: Mapping[str, Any], *, official: bool) -> dict[str, Any]:
    """Call the inherited durable JSONL primitive with its exact API signature."""
    return offset._append_ledger(event, path=path, official=official)


def _assert_append_prefix(path: Path, freeze_data: Mapping[str, Any]) -> None:
    prefix_bytes = int(freeze_data["pre_reservation_ledger_bytes"])
    with path.open("rb") as stream:
        prefix = stream.read(prefix_bytes)
    if (len(prefix) != prefix_bytes
            or hashlib.sha256(prefix).hexdigest() != freeze_data["pre_reservation_ledger_sha256"]):
        raise MomentsPilotError("Append-only shared ledger prefix differs from prefit freeze")


def reserve() -> dict[str, Any]:
    freeze_data = _load_freeze()
    if _git("status", "--porcelain"):
        raise MomentsPilotError("Commit the source freeze before appending a live budget reservation")
    events, snapshot = _ledger_snapshot(LEDGER)
    _assert_append_prefix(LEDGER, freeze_data)
    official = LEDGER.resolve() == offset.LEDGER.resolve()
    if not official:
        raise MomentsPilotError("Configured official shared ledger path mismatch")
    _append_pilot_reservation(LEDGER, freeze_data, official=True)
    events_after, snapshot_after = _ledger_snapshot(LEDGER)
    if snapshot_after != NEW_PRE_FIT_SNAPSHOT or len(events_after) != 348:
        raise MomentsPilotError(f"Append-only POS_MOMENTS reservation failed independent snapshot: {snapshot_after}")
    return _write_reservation_report(freeze_data)


def _append_pilot_reservation(path: Path, freeze_data: Mapping[str, Any], *, official: bool) -> None:
    # Support exact replay after a durable reservation while refusing any partial
    # amendment or unrelated append. The prefix is immutable in both states.
    events, snapshot = _ledger_snapshot(path)
    _assert_append_prefix(path, freeze_data)
    prefix_rows = int(freeze_data["pre_reservation_ledger_rows"])
    amendment_rows = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservation_rows = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(events) == prefix_rows:
        if snapshot != BASE_SNAPSHOT or any(e.get("card_id") == CARD_ID for e in events):
            raise MomentsPilotError("Budget amendment path differs from the exact frozen base ledger")
        append_reservation = False
    elif len(events) == prefix_rows + 1:
        amendment = amendment_rows[0] if len(amendment_rows) == 1 else {}
        if (snapshot != AMENDED_PRE_RESERVATION_SNAPSHOT or len(amendment_rows) != 1 or reservation_rows
                or amendment.get("prior_global_cap") != 137 or amendment.get("global_cap") != 142
                or amendment.get("prior_effective_caps") != BASE_CAPS or amendment.get("effective_caps") != NEW_CAPS
                or amendment.get("fit_ids") != list(FIT_IDS)
                or amendment.get("pre_amendment_ledger_sha256") != freeze_data["pre_reservation_ledger_sha256"]
                or amendment.get("source_commit") != freeze_data["source_commit"]
                or amendment.get("freeze_signature") != freeze_data["freeze_signature"]):
            raise MomentsPilotError("Single-event cap amendment is not the exact recoverable POS_MOMENTS event")
        append_reservation = True
    else:
        if len(events) != prefix_rows + 2 or len(amendment_rows) != 1 or len(reservation_rows) != 1:
            raise MomentsPilotError("Partial, duplicated, or unrelated POS_MOMENTS reservation append")
        amendment, reservation = amendment_rows[0], reservation_rows[0]
        if (snapshot != NEW_PRE_FIT_SNAPSHOT
                or amendment.get("pre_amendment_ledger_sha256") != freeze_data["pre_reservation_ledger_sha256"]
                or amendment.get("prior_global_cap") != 137 or amendment.get("global_cap") != 142
                or amendment.get("prior_effective_caps") != BASE_CAPS or amendment.get("effective_caps") != NEW_CAPS
                or amendment.get("source_commit") != freeze_data["source_commit"]
                or amendment.get("freeze_signature") != freeze_data["freeze_signature"]
                or amendment.get("fit_ids") != list(FIT_IDS)
                or reservation.get("fit_ids") != list(FIT_IDS)
                or reservation.get("source_commit") != freeze_data["source_commit"]
                or reservation.get("freeze_signature") != freeze_data["freeze_signature"]):
            raise MomentsPilotError("Existing POS_MOMENTS reservation does not match the exact frozen authorization")
        return
    if not append_reservation:
        _append_event(path, {
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
            "prior_global_cap": 137, "global_cap": 142,
            "prior_effective_caps": dict(BASE_CAPS), "effective_caps": dict(NEW_CAPS),
            "fit_ids": list(FIT_IDS), "source_commit": freeze_data["source_commit"],
            "freeze_signature": freeze_data["freeze_signature"],
            "pre_amendment_ledger_sha256": freeze_data["pre_reservation_ledger_sha256"],
            "pre_amendment_snapshot": snapshot,
            "reason": "Authorized POS_MOMENTS_V1 four-fit pilot; one extra continuation-cap slot remains unreserved and conditional on separate Sol review.",
        }, official=official)
    _append_event(path, {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
        "reservation_key": RESERVATION_KEY, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "global_cap": 142, "effective_category_cap": 88,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
    }, official=official)
    events_after = offset._ledger_events(path)
    if len(events_after) != 348 or offset.ledger_snapshot(events_after) != NEW_PRE_FIT_SNAPSHOT:
        raise MomentsPilotError("Temporary/live budget amendment did not reconstruct exact post-reservation snapshot")


def _write_reservation_report(freeze_data: Mapping[str, Any]) -> dict[str, Any]:
    events, snapshot = _ledger_snapshot(LEDGER)
    if snapshot != NEW_PRE_FIT_SNAPSHOT:
        raise MomentsPilotError(f"Live post-reservation budget differs: {snapshot}")
    amendment = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendment) != 1 or len(reservation) != 1:
        raise MomentsPilotError("Expected one cap amendment and one four-fit pilot reservation")
    body = {
        "status": "RESERVED_4_PILOT_FITS", "card_id": CARD_ID,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "ledger_sha256": sha256_file(LEDGER), "ledger_snapshot": snapshot,
        "amendment_event": amendment[0], "reservation_event": reservation[0],
        "fit_order": list(FIT_IDS), "fit_count": 4,
        "conditional_e79_fit_id": f"{CARD_ID}_E79_REAL",
        "conditional_e79_reserved": False, "conditional_e79_start_authorized": False,
        "conditional_reason": "Requires pilot outcome review and separate Sol decision; no E79 slot reused from retired contrast.",
    }
    _atomic_json(RESERVATION_REPORT, body)
    _atomic_json(RUN_ROOT / "RESERVATION.json", body)
    return body


def _check_live_ledger(
    freeze_data: Mapping[str, Any], *, allow_pending: str | None = None,
    ledger_path: Path = LEDGER, run_root: Path = RUN_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _assert_append_prefix(ledger_path, freeze_data)
    events, snapshot = _ledger_snapshot(ledger_path)
    amendment = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendment) != 1 or len(reservations) != 1:
        raise MomentsPilotError("Live ledger lacks one exact POS_MOMENTS amendment/reservation")
    if (amendment[0].get("card_id") != CARD_ID
            or amendment[0].get("prior_global_cap") != 137 or amendment[0].get("global_cap") != 142
            or amendment[0].get("prior_effective_caps") != BASE_CAPS
            or amendment[0].get("effective_caps") != NEW_CAPS
            or amendment[0].get("fit_ids") != list(FIT_IDS)
            or amendment[0].get("pre_amendment_ledger_sha256") != freeze_data["pre_reservation_ledger_sha256"]
            or amendment[0].get("source_commit") != freeze_data["source_commit"]
            or amendment[0].get("freeze_signature") != freeze_data["freeze_signature"]
            or reservations[0].get("card_id") != CARD_ID
            or reservations[0].get("reservation_key") != RESERVATION_KEY
            or reservations[0].get("amendment_id") != AMENDMENT_ID
            or reservations[0].get("fits") != 4 or reservations[0].get("category") != "pilot"
            or reservations[0].get("effective_category_cap") != 88 or reservations[0].get("global_cap") != 142
            or reservations[0].get("fit_ids") != list(FIT_IDS)
            or reservations[0].get("source_commit") != freeze_data["source_commit"]
            or reservations[0].get("freeze_signature") != freeze_data["freeze_signature"]):
        raise MomentsPilotError("Live budget events are not bound to the frozen source/fit IDs")
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    started_ids = [e["fit_id"] for e in starts]
    if len(started_ids) != len(set(started_ids)) or started_ids != list(FIT_IDS[:len(started_ids)]):
        raise MomentsPilotError("POS_MOMENTS FIT_STARTED sequence is duplicated or out of frozen order")
    terminal_ids = [e["fit_id"] for e in [*saved, *failed]]
    if len(terminal_ids) != len(set(terminal_ids)) or not set(terminal_ids).issubset(started_ids):
        raise MomentsPilotError("POS_MOMENTS terminal fit accounting is inconsistent")
    pending = [fit_id for fit_id in started_ids if fit_id not in terminal_ids]
    if len(pending) > 1 or (allow_pending is None and pending) or (allow_pending is not None and any(x != allow_pending for x in pending)):
        raise MomentsPilotError(f"Unexpected live pending fit: {pending}")
    expected = dict(NEW_PRE_FIT_SNAPSHOT)
    expected["event_rows"] += len(starts) + len(saved) + len(failed)
    expected["fit_starts"] += len(starts)
    expected["fit_saved"] += len(saved)
    expected["fit_failed"] += len(failed)
    expected["fit_pending"] = len(pending)
    if snapshot != expected:
        raise MomentsPilotError(f"Shared live ledger/accounting drifted: {snapshot}; expected {expected}")
    for event in [*starts, *saved, *failed]:
        if (event.get("reservation_key") != RESERVATION_KEY or event.get("card_id") != CARD_ID
                or event.get("amendment_id") != AMENDMENT_ID
                or event.get("freeze_signature") != freeze_data["freeze_signature"]
                or event.get("source_commit") != freeze_data["source_commit"]):
            raise MomentsPilotError("A POS_MOMENTS fit ledger event is not bound to its frozen reservation")
    start_by_id = {event["fit_id"]: event for event in starts}
    for event in saved:
        fit_id = event["fit_id"]
        artifact_path, receipt_path = _atomic_fit_artifacts(run_root, fit_id)
        receipt = _fit_receipt_valid(artifact_path, receipt_path, freeze_data=freeze_data, fit_id=fit_id)
        if (event.get("receipt_sha256") != sha256_file(receipt_path)
                or receipt.get("fit_start_event_sha256") != _sha_obj(start_by_id[fit_id])):
            raise MomentsPilotError(f"Saved {fit_id} FIT_SAVED event or start receipt hash mismatch")
    return events, snapshot


def _fit_rows_target(fit_rows: np.ndarray, fit_ids: np.ndarray, expected_sha: str) -> np.ndarray:
    rows = np.asarray(fit_rows, dtype=np.int64)
    ids_expected = np.asarray(fit_ids).astype(str)
    if len(rows) != len(ids_expected) or len(np.unique(rows)) != len(rows):
        raise MomentsPilotError("Frozen fit row/ID arrays are invalid")
    index = {int(row): i for i, row in enumerate(rows.tolist())}
    target = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or "record_id" not in header or "artifact_signal" not in header:
            raise MomentsPilotError("Official Train target header is invalid")
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, row in enumerate(reader):
            output_index = index.get(row_index)
            if output_index is None:
                continue
            if max(id_col, target_col) >= len(row) or row[id_col] != ids_expected[output_index]:
                raise MomentsPilotError("Fit-prefix target identity does not match frozen row IDs")
            target[output_index] = int(row[target_col])
            seen[output_index] = True
    if not seen.all() or not np.isin(target, (0, 1)).all() or sha256_array(target) != expected_sha:
        raise MomentsPilotError("Fit-prefix labels differ from signed parent target hash")
    return target


def _score_rows_target(score_rows: np.ndarray, score_ids: np.ndarray, score_cycles: np.ndarray) -> np.ndarray:
    """Read artifact_signal only for preverified temporal query row positions."""
    rows = np.asarray(score_rows, dtype=np.int64)
    ids = np.asarray(score_ids).astype(str)
    cycles_expected = np.asarray(score_cycles, dtype=np.int64)
    if len(rows) != len(ids) or len(rows) != len(cycles_expected) or len(np.unique(rows)) != len(rows):
        raise MomentsPilotError("Frozen query row/ID/cycle arrays are invalid")
    index = {int(row): i for i, row in enumerate(rows.tolist())}
    target = np.empty(len(rows), dtype=np.int8)
    seen = np.zeros(len(rows), dtype=bool)
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or not {"record_id", "cycle_code", "artifact_signal"}.issubset(header):
            raise MomentsPilotError("Official Train score target header is invalid")
        id_col, cycle_col, target_col = (header.index("record_id"), header.index("cycle_code"), header.index("artifact_signal"))
        for row_index, row in enumerate(reader):
            output_index = index.get(row_index)
            if output_index is None:
                continue
            if (max(id_col, cycle_col, target_col) >= len(row)
                    or row[id_col] != ids[output_index]
                    or int(row[cycle_col]) != cycles_expected[output_index]):
                raise MomentsPilotError("Temporal query target identity/cycle does not match verified prediction rows")
            target[output_index] = int(row[target_col])
            seen[output_index] = True
    if not seen.all() or not np.isin(target, (0, 1)).all():
        raise MomentsPilotError("Temporal query labels are missing or non-binary")
    return target


def _moment_cache(origin: int, arm: str, part: str) -> sparse.csr_matrix:
    path = _cache_path(origin, f"{part}_{arm.lower()}")
    if not path.is_file():
        raise MomentsPilotError(f"Frozen {part}_{arm.lower()} matrix cache is missing")
    return sparse.load_npz(path).tocsr().astype(np.float64)


def _atomic_fit_artifacts(root: Path, fit_id: str) -> tuple[Path, Path]:
    return root / "units" / f"{fit_id}.npz", root / "receipts" / f"{fit_id}.json"


def _fit_receipt_valid(artifact_path: Path, receipt_path: Path, *, freeze_data: Mapping[str, Any], fit_id: str) -> dict[str, Any]:
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise MomentsPilotError(f"Saved {fit_id} gamma artifact or receipt is missing")
    receipt = _read_json(receipt_path)
    if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != freeze_data["freeze_signature"]
            or receipt.get("source_commit") != freeze_data["source_commit"]
            or receipt.get("source_hashes") != freeze_data.get("source_hashes")
            or receipt.get("input_sha256") != freeze_data.get("input_sha256")
            or receipt.get("artifact_sha256") != sha256_file(artifact_path)):
        raise MomentsPilotError(f"Saved {fit_id} gamma receipt binding mismatch")
    gamma = _atomic_load_npz(artifact_path, ["gamma"])["gamma"].astype(np.float64)
    optimizer = receipt.get("optimizer", {})
    if (sha256_array(gamma) != receipt.get("gamma_sha256") or not np.isfinite(gamma).all()
            or optimizer.get("success") is not True or optimizer.get("c") != 1.0
            or optimizer.get("options") != {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 50}):
        raise MomentsPilotError(f"Saved {fit_id} gamma or optimizer metadata is invalid")
    return receipt


def _assert_active_freeze(callback_signature: str | None, freeze_data: Mapping[str, Any]) -> None:
    if callback_signature is None or callback_signature != freeze_data.get("freeze_signature"):
        raise MomentsPilotError("Callback freeze signature differs from the active successor freeze")


def _resume_durable_fit(
    fit_id: str,
    *,
    freeze_data: Mapping[str, Any],
    ledger_path: Path,
    artifact_path: Path,
    receipt_path: Path,
    official: bool,
) -> str:
    _assert_active_freeze(freeze_data.get("freeze_signature"), freeze_data)
    if official and freeze_data.get("freeze_signature") != _load_freeze().get("freeze_signature"):
        raise MomentsPilotError("Resume callback freeze signature does not match active source freeze")
    events = offset._ledger_events(ledger_path)
    started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(started) != 1 or len(terminal) > 1:
        raise MomentsPilotError(f"Resume state for {fit_id} is inconsistent")
    if terminal:
        if terminal[0]["event"] == "FIT_FAILED":
            return "failed_consumed_no_retry"
        receipt = _fit_receipt_valid(artifact_path, receipt_path, freeze_data=freeze_data, fit_id=fit_id)
        if terminal[0].get("receipt_sha256") != sha256_file(receipt_path):
            raise MomentsPilotError(f"Saved {fit_id} terminal receipt hash mismatch")
        return "already_saved"
    if not artifact_path.is_file() or not receipt_path.is_file():
        failure = {
            "fit_id": fit_id, "source_commit": freeze_data["source_commit"],
            "freeze_signature": freeze_data["freeze_signature"],
            "error": "started fit interrupted before both gamma artifact and receipt were durable; consumed/no retry",
            "fit_consumed_no_retry": True,
        }
        failure_path = artifact_path.parent.parent / "failures" / f"{fit_id}.json"
        _atomic_json(failure_path, failure)
        _append_event(ledger_path, {
            "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": 67 if "E67" in fit_id else 75,
            "fit_kind": "PREDICTOR", "amendment_id": AMENDMENT_ID,
            "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
            "error": failure["error"], "fit_consumed_no_retry": True,
        }, official=official)
        return "failed_interrupted_no_retry"
    receipt = _fit_receipt_valid(artifact_path, receipt_path, freeze_data=freeze_data, fit_id=fit_id)
    start = started[0]
    _append_event(ledger_path, {
        "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": receipt["origin"], "fit_kind": "PREDICTOR",
        "receipt_sha256": sha256_file(receipt_path), "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
        "recovered_without_refit": True, "started_event_sha256": _sha_obj(start),
    }, official=official)
    return "recovered_saved_without_refit"


def _run_one_fit(
    fit_id: str,
    *,
    origin: int,
    arm: str,
    x: sparse.spmatrix,
    z0: np.ndarray,
    y: np.ndarray,
    fit_rows: np.ndarray,
    fit_ids: np.ndarray,
    freeze_data: Mapping[str, Any],
    ledger_path: Path,
    run_root: Path,
    official: bool,
    interrupt_after_durable: bool = False,
    resource: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if fit_id not in FIT_IDS or fit_id != f"{CARD_ID}_E{origin}_{arm}":
        raise MomentsPilotError("Fit ID/origin/arm is outside frozen order")
    if freeze_data.get("freeze_signature") is None:
        raise MomentsPilotError("Fit callback lacks active freeze signature")
    _assert_active_freeze(freeze_data.get("freeze_signature"), freeze_data)
    if official and freeze_data["freeze_signature"] != _load_freeze()["freeze_signature"]:
        raise MomentsPilotError("Fit callback received a stale/wrong freeze signature")
    artifact_path, receipt_path = _atomic_fit_artifacts(run_root, fit_id)
    current_events = offset._ledger_events(ledger_path)
    started = [e for e in current_events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in current_events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if terminal:
        status = _resume_durable_fit(fit_id, freeze_data=freeze_data, ledger_path=ledger_path,
                                     artifact_path=artifact_path, receipt_path=receipt_path, official=official)
        return {"fit_id": fit_id, "status": status, "receipt_sha256": sha256_file(receipt_path) if receipt_path.exists() else None}
    if started:
        status = _resume_durable_fit(fit_id, freeze_data=freeze_data, ledger_path=ledger_path,
                                     artifact_path=artifact_path, receipt_path=receipt_path, official=official)
        return {"fit_id": fit_id, "status": status, "receipt_sha256": sha256_file(receipt_path) if receipt_path.exists() else None}
    if x.shape[0] != len(y) or len(z0) != len(y) or len(fit_rows) != len(y) or len(fit_ids) != len(y):
        raise MomentsPilotError(f"{fit_id} fit arrays/labels/population do not align")
    if x.shape[1] <= 0 or not np.isfinite(x.data).all() or not np.isfinite(z0).all() or not np.isin(y, (0, 1)).all():
        raise MomentsPilotError(f"{fit_id} fit inputs are invalid")
    if resource is None:
        resource = _check_fit_resources(estimate_bytes=1024**3, initial=False)
    events, snapshot = _check_live_ledger(freeze_data) if official else _ledger_snapshot(ledger_path)
    if official and snapshot["fit_pending"] != 0:
        raise MomentsPilotError("Another shared fit is pending; defer without consuming this ID")
    if official:
        own_starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
        if len(own_starts) >= 4 or fit_id != FIT_IDS[len(own_starts)]:
            raise MomentsPilotError("Next requested fit is not the next unused frozen ID")
    row_meta = {
        "rows": len(y), "fit_rows_sha256": sha256_array(fit_rows), "fit_ids_sha256": sha256_array(np.asarray(fit_ids).astype(str)),
        "fit_target_sha256": sha256_array(y), "fit_target_read_purpose": "origin-prefix-fit-only",
        "origin": origin, "arm": arm, "matrix_sha256": sha256_csr(x), "offset_sha256": sha256_array(z0),
    }
    _append_event(ledger_path, {
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": f"moments_{arm.lower()}",
        "fit_kind": "PREDICTOR", "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
        "fit_population": row_meta, "resource_snapshot": dict(resource),
    }, official=official)
    started_at = time.perf_counter()
    try:
        with threadpool_limits(limits=THREADS):
            gamma, optimizer = fit_offset_coefficients(x, z0, y, c=1.0)
        gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
        if len(gamma) != x.shape[1] or not np.isfinite(gamma).all():
            raise MomentsPilotError("Optimizer gamma shape or finiteness check failed")
        _atomic_npz(artifact_path, gamma=gamma)
        started_events = [event for event in offset._ledger_events(ledger_path)
                          if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
        if len(started_events) != 1:
            raise MomentsPilotError(f"Expected exactly one durable FIT_STARTED event for {fit_id}")
        receipt = {
            "schema_version": 1, "fit_id": fit_id, "card_id": CARD_ID, "origin": origin, "arm": arm,
            "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
            "source_hashes": freeze_data["source_hashes"], "input_sha256": freeze_data["input_sha256"],
            "fit_population": row_meta,
            "matrix": {"shape": list(x.shape), "sha256": sha256_csr(x), "offset_sha256": sha256_array(z0)},
            "optimizer": optimizer, "gamma_sha256": sha256_array(gamma),
            "artifact_path": artifact_path.relative_to(ROOT).as_posix() if official else str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path), "resource_before": dict(resource),
            "resource_after": _resource_snapshot(0), "elapsed_seconds": float(time.perf_counter() - started_at),
            "fit_start_event_sha256": _sha_obj(started_events[0]),
        }
        _atomic_json(receipt_path, receipt)
        if sha256_file(artifact_path) != receipt["artifact_sha256"]:
            raise MomentsPilotError("Gamma artifact hash changed during durable save")
        if interrupt_after_durable:
            return {"fit_id": fit_id, "status": "durable_pending_for_synthetic_resume",
                    "artifact_sha256": receipt["artifact_sha256"], "receipt_sha256": sha256_file(receipt_path)}
        _append_event(ledger_path, {
            "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
            "receipt_sha256": sha256_file(receipt_path), "amendment_id": AMENDMENT_ID,
            "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
        }, official=official)
        if official:
            _check_live_ledger(freeze_data)
        return {"fit_id": fit_id, "status": "saved", "optimizer": optimizer,
                "gamma_sha256": receipt["gamma_sha256"], "receipt_sha256": sha256_file(receipt_path)}
    except Exception as exc:
        now = offset._ledger_events(ledger_path)
        started = [e for e in now if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
        terminals = [e for e in now if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
        if started and not terminals:
            if artifact_path.is_file() and receipt_path.is_file():
                try:
                    _fit_receipt_valid(artifact_path, receipt_path, freeze_data=freeze_data, fit_id=fit_id)
                except Exception:
                    pass
                else:
                    # The optimizer output is durable. Leave the ledger pending
                    # so the next invocation can publish FIT_SAVED without refit.
                    raise
            failure_path = run_root / "failures" / f"{fit_id}.json"
            _atomic_json(failure_path, {
                "fit_id": fit_id, "origin": origin, "arm": arm,
                "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
                "error_type": type(exc).__name__, "error": str(exc),
                "fit_consumed_no_retry": True,
            })
            _append_event(ledger_path, {
                "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
                "amendment_id": AMENDMENT_ID, "freeze_signature": freeze_data["freeze_signature"],
                "source_commit": freeze_data["source_commit"], "error": f"{type(exc).__name__}: {exc}",
                "fit_consumed_no_retry": True,
            }, official=official)
        raise


def _lock_fit_run():
    import contextlib
    import msvcrt
    lock_path = RUN_ROOT / "fit.lock"

    @contextlib.contextmanager
    def manager():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
        unlock = False
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                unlock = True
            except OSError as exc:
                raise MomentsPilotError("Another POS_MOMENTS process holds the fit lock") from exc
            payload = _canonical_bytes({"pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()})
            stream.seek(0)
            stream.write(b"\0" + payload)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            if unlock:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            stream.close()
    return manager()


def run_fits() -> list[dict[str, Any]]:
    freeze_data = _load_freeze()
    reservation = _read_json(RESERVATION_REPORT)
    if reservation.get("status") != "RESERVED_4_PILOT_FITS" or reservation.get("freeze_signature") != freeze_data["freeze_signature"]:
        raise MomentsPilotError("Four-fit reservation is not bound to the current freeze")
    initial_events, _ = _ledger_snapshot()
    initial_starts = [e for e in initial_events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    initial_terminals = {e.get("fit_id") for e in initial_events
                         if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") in FIT_IDS}
    initial_pending = [e["fit_id"] for e in initial_starts if e["fit_id"] not in initial_terminals]
    allowed_pending = initial_pending[0] if len(initial_pending) == 1 else None
    _check_live_ledger(freeze_data, allow_pending=allowed_pending)
    results: list[dict[str, Any]] = []
    previous_origin_state: dict[str, Any] | None = None
    current_origin: int | None = None
    with _lock_fit_run():
        for order_index, fit_id in enumerate(FIT_IDS):
            origin = 67 if "E67" in fit_id else 75
            arm = "REAL" if fit_id.endswith("_REAL") else "SHUFFLED"
            events, _ = _ledger_snapshot()
            started_ids = [event["fit_id"] for event in events
                           if event.get("event") == "FIT_STARTED" and event.get("fit_id") in FIT_IDS]
            terminal_ids = {event.get("fit_id") for event in events
                            if event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and event.get("fit_id") in FIT_IDS}
            pending_ids = [value for value in started_ids if value not in terminal_ids]
            allow_pending = pending_ids[0] if len(pending_ids) == 1 else None
            events, _ = _check_live_ledger(freeze_data, allow_pending=allow_pending)
            existing = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
            existing_terminal = [event for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and event.get("fit_id") == fit_id]
            if existing and not existing_terminal:
                artifact, receipt = _atomic_fit_artifacts(RUN_ROOT, fit_id)
                status = _resume_durable_fit(fit_id, freeze_data=freeze_data, ledger_path=LEDGER,
                                             artifact_path=artifact, receipt_path=receipt, official=True)
                results.append({"fit_id": fit_id, "status": status})
                if status in {"failed_consumed_no_retry"}:
                    continue
                _check_live_ledger(freeze_data)
                continue
            if existing_terminal:
                results.append({"fit_id": fit_id, "status": "already_saved" if existing_terminal[0]["event"] == "FIT_SAVED" else "failed_consumed_no_retry"})
                continue
            if current_origin != origin:
                del previous_origin_state
                gc.collect()
                frame = _identity_features()
                previous_origin_state = _load_parent_origin(origin, frame)
                current_origin = origin
            state = previous_origin_state
            if state is None:
                raise MomentsPilotError("Origin source state was not loaded")
            fit_matrix = _moment_cache(origin, arm, "fit")
            expected_fit = freeze_data["origins"][f"E{origin}"]["moments"][arm.lower()]["fit"]
            if (fit_matrix.shape != tuple(expected_fit["shape"])
                    or sha256_csr(fit_matrix) != expected_fit["sha256"]):
                raise MomentsPilotError(f"E{origin} {arm} frozen moment fit matrix changed")
            z0 = np.load(RUN_ROOT / f"E{origin}/z0_fit.npy", allow_pickle=False)
            y = _fit_rows_target(state["fit_rows"], state["fit_ids"], state["bag_target_sha256"])
            if sha256_array(z0) != freeze_data["origins"][f"E{origin}"]["z0_fit_sha256"]:
                raise MomentsPilotError(f"E{origin} frozen BAG_LR offset changed before fit")
            if sha256_file(TRAIN) != freeze_data["input_sha256"]:
                raise MomentsPilotError("Official Train changed immediately before fit")
            memory_estimate = _fit_memory_estimate(fit_matrix)["estimated_incremental_bytes"]
            resource = _check_fit_resources(estimate_bytes=memory_estimate, initial=(order_index == 0))
            if resource["process_rss_bytes"] > MAX_RSS:
                raise MomentsPilotError(f"RESOURCE_BLOCK_BEFORE_FIT: process RSS exceeds 6 GiB: {resource['process_rss_bytes']}")
            result = _run_one_fit(fit_id, origin=origin, arm=arm, x=fit_matrix, z0=z0, y=y,
                                  fit_rows=state["fit_rows"], fit_ids=state["fit_ids"],
                                  freeze_data=freeze_data, ledger_path=LEDGER, run_root=RUN_ROOT,
                                  official=True, resource=resource)
            results.append(result)
            if result["status"] in {"failed_consumed_no_retry"}:
                continue
            del fit_matrix, z0, y
            gc.collect()
    _atomic_json(RUN_ROOT / "FIT_RESULTS.json", {"fit_ids": list(FIT_IDS), "results": results,
                                                  "freeze_signature": freeze_data["freeze_signature"]})
    return results


def build_predictions() -> dict[str, Any]:
    freeze_data = _load_freeze()
    events, snapshot = _check_live_ledger(freeze_data)
    saved_ids = [e["fit_id"] for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed_ids = [e["fit_id"] for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    if saved_ids != [fit_id for fit_id in FIT_IDS if fit_id not in failed_ids] or failed_ids:
        raise MomentsPilotError(f"All four fits must be durable before prediction creation; saved={saved_ids}, failed={failed_ids}")
    if snapshot["fit_pending"] != 0:
        raise MomentsPilotError("Prediction creation blocked by pending fit")
    frame = _identity_features()
    bundles: dict[str, Any] = {}
    for origin in ORIGINS:
        state = _load_parent_origin(origin, frame)
        z0_score = np.load(RUN_ROOT / f"E{origin}/z0_score.npy", allow_pickle=False)
        if sha256_array(z0_score) != freeze_data["origins"][f"E{origin}"]["z0_score_sha256"]:
            raise MomentsPilotError(f"E{origin} frozen BAG_LR score offset changed")
        score_data = {
            "score_rows": state["score_rows"], "score_record_ids": state["score_ids"],
            "score_cycles": state["score_cycles"], "present_mask": state["present"],
            "entry_friction": state["entry_friction"], "text_load_group": state["text_load_group"],
            "prob_pP": state["p_old"], "prob_a02": state["p_a02"],
            "prob_bag": np.asarray(state["prediction"]["prob_bag"], dtype=np.float64),
        }
        present = score_data["present_mask"].astype(bool)
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            artifact, receipt_path = _atomic_fit_artifacts(RUN_ROOT, fit_id)
            receipt = _fit_receipt_valid(artifact, receipt_path, freeze_data=freeze_data, fit_id=fit_id)
            gamma = _atomic_load_npz(artifact, ["gamma"])["gamma"].astype(np.float64)
            if sha256_array(gamma) != receipt.get("gamma_sha256"):
                raise MomentsPilotError(f"E{origin} {arm} gamma bytes differ from receipt")
            moment_score = _moment_cache(origin, arm, "score")
            expected = freeze_data["origins"][f"E{origin}"]["moments"][arm.lower()]["score"]
            if (moment_score.shape != tuple(expected["shape"]) or sha256_csr(moment_score) != expected["sha256"]):
                raise MomentsPilotError(f"E{origin} {arm} score moments differ from frozen preflight")
            if np.asarray(z0_score).shape[0] != moment_score.shape[0] or gamma.shape[0] != moment_score.shape[1]:
                raise MomentsPilotError(f"E{origin} {arm} score prediction dimensions do not align")
            q = offset_probability(moment_score, z0_score, gamma)
            candidate, mix50 = fixed_blends(q, score_data["prob_a02"], score_data["prob_pP"], present)
            score_data[f"prob_q_{arm.lower()}"] = q
            score_data[f"prob_candidate_{arm.lower()}"] = candidate
            score_data[f"fit_receipt_sha256_{arm.lower()}"] = np.asarray([sha256_file(receipt_path)])
        score_data["prob_mix50"] = mix50
        if not np.array_equal(score_data["prob_candidate_real"][~present], score_data["prob_pP"][~present]):
            raise MomentsPilotError(f"E{origin} real candidate missing route differs from pP")
        if not np.array_equal(score_data["prob_candidate_shuffled"][~present], score_data["prob_pP"][~present]):
            raise MomentsPilotError(f"E{origin} shuffled control missing route differs from pP")
        score_data = {key: np.asarray(value) for key, value in score_data.items()}
        bundle_path = RUN_ROOT / f"E{origin}/predictions_E{origin}.npz"
        _atomic_npz(bundle_path, **score_data)
        bundles[f"E{origin}"] = {
            "path": bundle_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(bundle_path),
            "rows": len(state["score_rows"]), "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
            "score_record_ids_sha256": sha256_array(state["score_ids"]),
            "score_cycles_sha256": sha256_array(state["score_cycles"]),
        }
    ready = {
        "schema_version": 1, "card_id": CARD_ID, "status": "PASS_PREDICTIONS_DURABLE_PRELABEL_CLOSED",
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "fit_ids": list(FIT_IDS), "fit_failures": [], "predictions": bundles,
        "query_labels_read": False, "test_or_sample_access": False,
        "policy": ".5*q+.5*A02 on present; original pP exactly on missing; threshold>=.50",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    _atomic_json(REPORT_ROOT / "PREDICTIONS_READY.json", ready)
    return ready


def verify_prelabel() -> dict[str, Any]:
    freeze_data = _load_freeze()
    from scripts.verify_optimization_v3_positional_moments_v1 import verify_prelabel_outputs
    result = verify_prelabel_outputs(ROOT, RUN_ROOT, freeze_data)
    if result.get("status") != "PASS" or result.get("query_labels_read") is not False:
        raise MomentsPilotError(f"Independent prelabel verification failed: {result}")
    _atomic_json(RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
    _atomic_json(REPORT_ROOT / "PRELABEL_VERIFICATION.json", result)
    return result


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, confusion_matrix, f1_score,
        log_loss, precision_score, recall_score, roc_auc_score,
    )
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(target) != len(p) or len(target) == 0 or not np.isin(target, (0, 1)).all():
        raise MomentsPilotError("Metric target/probability rows are invalid")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise MomentsPilotError("Metric probabilities are non-finite or outside [0,1]")
    pred = p >= 0.50
    cm = confusion_matrix(target, pred.astype(np.int8), labels=[0, 1])
    two_classes = len(np.unique(target)) == 2
    return {
        "n": int(len(target)), "negative": int((target == 0).sum()), "positive": int((target == 1).sum()),
        "macro_f1": float(f1_score(target, pred.astype(np.int8), average="macro", zero_division=0)),
        "positive_f1": float(f1_score(target, pred.astype(np.int8), pos_label=1, zero_division=0)),
        "precision": float(precision_score(target, pred.astype(np.int8), pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(target, pred.astype(np.int8), pos_label=1, zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "roc_auc": float(roc_auc_score(target, p)) if two_classes else None,
        "average_precision": float(average_precision_score(target, p)) if two_classes else None,
        "log_loss": float(log_loss(target, p, labels=[0, 1])),
        "brier": float(brier_score_loss(target, p)),
    }


def _views(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {
            "H12_E67": (cycles >= 68) & (cycles <= 79),
            "W4": (cycles >= 68) & (cycles <= 71),
            "W5": (cycles >= 72) & (cycles <= 75),
            "W6": (cycles >= 76) & (cycles <= 79),
        }
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def _score_origin(origin: int, bundle: Mapping[str, np.ndarray], y: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cycles = np.asarray(bundle["score_cycles"], dtype=np.int64)
    present = np.asarray(bundle["present_mask"], dtype=bool)
    entry = np.asarray(bundle["entry_friction"], dtype=np.int8)
    load_group = np.asarray(bundle["text_load_group"], dtype=np.int8)
    policies = {
        "moment_real": np.asarray(bundle["prob_candidate_real"], dtype=np.float64),
        "moment_shuffled": np.asarray(bundle["prob_candidate_shuffled"], dtype=np.float64),
        "MIX50": np.asarray(bundle["prob_mix50"], dtype=np.float64),
        "q_real": np.asarray(bundle["prob_q_real"], dtype=np.float64),
        "q_shuffled": np.asarray(bundle["prob_q_shuffled"], dtype=np.float64),
        "pP_oldPOS": np.asarray(bundle["prob_pP"], dtype=np.float64),
        "A02": np.asarray(bundle["prob_a02"], dtype=np.float64),
        "bag": np.asarray(bundle["prob_bag"], dtype=np.float64),
    }
    if any(len(value) != len(y) for value in policies.values()) or len(cycles) != len(y):
        raise MomentsPilotError(f"E{origin} score rows do not align for metric calculation")
    view_results: dict[str, Any] = {}
    gate_checks: list[dict[str, Any]] = []
    subgroups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    group_masks: list[tuple[str, np.ndarray]] = [
        ("availability_present", present), ("availability_missing", ~present),
        ("entry_friction_0", entry == 0), ("entry_friction_1", entry == 1),
    ] + [(f"text_load_Q{i+1}", load_group == i) for i in range(4)]
    for view, view_mask in _views(origin, cycles).items():
        metrics = {name: _metric(y[view_mask], probability[view_mask]) for name, probability in policies.items()}
        deltas = {
            "real_minus_MIX50": metrics["moment_real"]["macro_f1"] - metrics["MIX50"]["macro_f1"],
            "real_minus_shuffled": metrics["moment_real"]["macro_f1"] - metrics["moment_shuffled"]["macro_f1"],
            "shuffled_minus_MIX50": metrics["moment_shuffled"]["macro_f1"] - metrics["MIX50"]["macro_f1"],
        }
        view_results[view] = {"n": int(view_mask.sum()), "metrics": metrics, "deltas": deltas}
        if origin == 67 and view in {"W4", "W5", "W6"}:
            gate_checks.append({
                "gate": f"E67_{view}_real_harm_vs_MIX50_max_0.003", "delta": deltas["real_minus_MIX50"],
                "limit": -0.003, "pass": bool(deltas["real_minus_MIX50"] >= -0.003 - GATE_TOLERANCE),
            })
        if (origin == 67 and view == "H12_E67") or (origin == 75 and view == "W6"):
            gate_checks.extend([
                {"gate": f"E{origin}_{view}_real_gain_vs_MIX50_min_0.0005", "delta": deltas["real_minus_MIX50"],
                 "limit": 0.0005, "pass": bool(deltas["real_minus_MIX50"] >= 0.0005 - GATE_TOLERANCE)},
                {"gate": f"E{origin}_{view}_real_gain_vs_shuffled_min_0", "delta": deltas["real_minus_shuffled"],
                 "limit": 0.0, "pass": bool(deltas["real_minus_shuffled"] >= -GATE_TOLERANCE)},
            ])
        for group_name, group_mask in group_masks:
            use = view_mask & group_mask
            n = int(use.sum())
            pos = int(np.asarray(y)[use].sum())
            neg = n - pos
            if n >= 200 and pos >= 30 and neg >= 30:
                real = _metric(np.asarray(y)[use], policies["moment_real"][use])
                mix = _metric(np.asarray(y)[use], policies["MIX50"][use])
                d_f1 = real["macro_f1"] - mix["macro_f1"]
                d_recall = real["positive_recall"] - mix["positive_recall"]
                passed = d_f1 >= -0.01 - GATE_TOLERANCE and d_recall >= -0.02 - GATE_TOLERANCE
                gate_checks.append({
                    "gate": f"E{origin}_{view}_{group_name}_guard", "delta_macro_f1": d_f1,
                    "delta_positive_recall": d_recall, "support": n, "positive": pos, "negative": neg,
                    "pass": bool(passed),
                })
                subgroups.append({"view": view, "group": group_name, "status": "SUPPORTED", "n": n,
                                  "positive": pos, "negative": neg, "real_macro_f1": real["macro_f1"],
                                  "mix50_macro_f1": mix["macro_f1"], "delta_macro_f1": d_f1,
                                  "real_positive_recall": real["positive_recall"],
                                  "mix50_positive_recall": mix["positive_recall"],
                                  "delta_positive_recall": d_recall})
            else:
                subgroups.append({"view": view, "group": group_name, "status": "UNKNOWN_UNSUPPORTED",
                                  "n": n, "positive": pos, "negative": neg})
        for cycle in sorted(np.unique(cycles[view_mask])):
            one = view_mask & (cycles == cycle)
            cycle_metric_real = _metric(np.asarray(y)[one], policies["moment_real"][one])
            cycle_metric_mix = _metric(np.asarray(y)[one], policies["MIX50"][one])
            per_cycle.append({"view": view, "cycle": int(cycle), "n": int(one.sum()),
                              "real_macro_f1": cycle_metric_real["macro_f1"],
                              "mix50_macro_f1": cycle_metric_mix["macro_f1"],
                              "delta": cycle_metric_real["macro_f1"] - cycle_metric_mix["macro_f1"]})
    return {"rows": len(y), "views": view_results, "subgroups": subgroups, "per_cycle": per_cycle}, gate_checks


def _plot_results(result: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    deltas: list[float] = []
    limits: list[float] = []
    for origin in ("E67", "E75"):
        for view in ("H12_E67", "W4", "W5", "W6") if origin == "E67" else ("W6",):
            data = result["origins"][origin]["views"][view]
            labels.append(f"{origin} {view}")
            deltas.append(float(data["deltas"]["real_minus_MIX50"]))
            limits.append(0.0005 if view in {"H12_E67", "W6"} and (origin == "E75" or view == "H12_E67") else -0.003)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    colors = ["#b64b4b" if value < limit else "#2e6f9e" for value, limit in zip(deltas, limits)]
    bars = ax.bar(np.arange(len(labels)), deltas, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=1)
    for bar, limit in zip(bars, limits):
        x = bar.get_x() + bar.get_width() / 2
        ax.plot([x - .30, x + .30], [limit, limit], color="#161616", linewidth=2.2)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=23, ha="right")
    ax.set_ylabel("Macro F1 delta: REAL blend − MIX50")
    ax.set_title("POS_MOMENTS_V1: fixed policy contrast across views")
    ax.grid(axis="y", alpha=.25)
    ax.text(.01, .02, "Black ticks mark each view's frozen minimum/harm boundary.", transform=ax.transAxes, fontsize=9)
    fig.savefig(REPORT_ROOT / "primary_gates.png", dpi=160)
    plt.close(fig)
    cycles = result["origins"]["E67"]["per_cycle"]
    e67 = [row for row in cycles if row["view"] == "H12_E67"]
    e75 = result["origins"]["E75"]["per_cycle"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    ax.plot([x["cycle"] for x in e67], [x["delta"] for x in e67], marker="o", label="E67 REAL − MIX50")
    ax.plot([x["cycle"] for x in e75], [x["delta"] for x in e75], marker="s", label="E75 REAL − MIX50")
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("cycle_code")
    ax.set_ylabel("Per-cycle Macro F1 delta")
    ax.set_title("Per-cycle sensitivity (E67/E75 overlap on cycles 76–79)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.savefig(REPORT_ROOT / "per_cycle_delta.png", dpi=160)
    plt.close(fig)


def _write_report(result: Mapping[str, Any], qa: Mapping[str, Any]) -> None:
    rows = [
        "# POS_MOMENTS_V1 — hasil pilot",
        "",
        f"**Status:** `{result['status']}`. Kandidat diagnostic; champion tidak berubah.",
        "",
        "## Perbandingan utama",
        "",
        "| View | n | REAL blend | MIX50 | Δ REAL−MIX50 | Shuffled blend | Δ REAL−shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for origin, view in (("E67", "H12_E67"), ("E67", "W4"), ("E67", "W5"), ("E67", "W6"), ("E75", "W6")):
        item = result["origins"][origin]["views"][view]
        met = item["metrics"]
        rows.append(f"| {origin} {view} | {item['n']} | {met['moment_real']['macro_f1']:.6f} | {met['MIX50']['macro_f1']:.6f} | {item['deltas']['real_minus_MIX50']:+.6f} | {met['moment_shuffled']['macro_f1']:.6f} | {item['deltas']['real_minus_shuffled']:+.6f} |")
    rows.extend(["", "## Gate summary", "", "| Gate | Result | Threshold | Status |", "|---|---:|---:|---|"])
    for gate in result["gate_checks"]:
        value = gate.get("delta", gate.get("delta_macro_f1"))
        limit = gate.get("limit", "F1 ≥ −0.01; recall ≥ −0.02")
        rows.append(f"| {gate['gate']} | {value:+.6f} | {limit} | {'PASS' if gate['pass'] else 'FAIL'} |")
    unknown = sum(item["status"] == "UNKNOWN_UNSUPPORTED"
                  for origin in result["origins"].values() for item in origin["subgroups"])
    rows.extend([
        "", f"Supported subgroup guards: {sum('subgroup_guard' in g['gate'] and g['pass'] for g in result['gate_checks'])} passed; unsupported subgroup views: {unknown} UNKNOWN.",
        "", "## Interpretasi dan batas", "",
        "W menggabungkan bag unigram B, momen posisi kontinu orde-1, dan momen kuadratik terpusat, lalu menormalisasi seluruh tiga kanal bersama-sama. Perbandingan adalah adaptif pada temporal Train; E67 dan E75 berbagi siklus 76–79. Hasil tidak membuktikan pengaruh kausal atau perbaikan public score.",
        "", "Prelabel QA independen: `PASS`. Test/Sample tidak dibuka. Tidak ada full-Train fit, upload, atau continuation E79.",
        "", f"Source commit: `{result['source_commit']}`; freeze signature: `{result['freeze_signature']}`.",
        f"Independent QA: `{qa['status']}`; result core SHA-256: `{qa['results_core_sha256']}`.",
        "", "![Gate deltas](primary_gates.png)", "", "![Per-cycle deltas](per_cycle_delta.png)", "",
    ])
    (REPORT_ROOT / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")


def score() -> dict[str, Any]:
    completions = [e for e in offset._ledger_events(LEDGER)
                   if e.get("event") == "CARD_COMPLETED" and e.get("card_id") == CARD_ID]
    if len(completions) > 1:
        raise MomentsPilotError("Duplicate POS_MOMENTS CARD_COMPLETED events")
    if completions:
        result_path = RUN_ROOT / "RESULTS.json"
        if not result_path.is_file() or completions[0].get("result_sha256") != sha256_file(result_path):
            raise MomentsPilotError("Existing completion/result is ambiguous; do not reopen labels")
        return _read_json(result_path)
    freeze_data = _load_freeze()
    prelabel = _read_json(RUN_ROOT / "PRELABEL_VERIFICATION.json")
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    ready = _read_json(ready_path)
    if (prelabel.get("status") != "PASS" or prelabel.get("query_labels_read") is not False
            or prelabel.get("freeze_signature") != freeze_data["freeze_signature"]
            or prelabel.get("predictions_ready_sha256") != sha256_file(ready_path)
            or ready.get("query_labels_read") is not False):
        raise MomentsPilotError("Outcome metrics remain closed until independent prelabel PASS")
    events, snapshot = _check_live_ledger(freeze_data)
    if (snapshot["fit_starts"] != 137 or snapshot["fit_saved"] != 131
            or snapshot["fit_failed"] != 6 or snapshot["fit_pending"] != 0):
        raise MomentsPilotError("Exactly four saved POS_MOMENTS fits are required before query label access")
    # First and only query-label access in this pilot: official Train temporal score rows.
    origins: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for origin in ORIGINS:
        bundle_path = ROOT / ready["predictions"][f"E{origin}"]["path"]
        if sha256_file(bundle_path) != ready["predictions"][f"E{origin}"]["sha256"]:
            raise MomentsPilotError(f"E{origin} prelabel prediction bundle changed before scoring")
        pred = _atomic_load_npz(bundle_path, [
            "score_rows", "score_record_ids", "score_cycles", "present_mask", "entry_friction", "text_load_group",
            "prob_pP", "prob_a02", "prob_bag", "prob_q_real", "prob_q_shuffled",
            "prob_candidate_real", "prob_candidate_shuffled", "prob_mix50",
        ])
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        ids = pred["score_record_ids"].astype(str)
        cycles = pred["score_cycles"].astype(np.int64)
        y = _score_rows_target(rows, ids, cycles)
        origin_result, gates = _score_origin(origin, pred, y)
        origins[f"E{origin}"] = {**origin_result, "query_labels_opened": True}
        all_gates.extend(gates)
    passed = bool(all(gate.get("pass") is True for gate in all_gates))
    core = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "PILOT_PASS_PENDING_SOL_REVIEW" if passed else "PILOT_GATE_FAILED",
        "qualified_candidate": False, "champion_change": False,
        "full_train_refit_authorized": False, "upload_authorized": False,
        "source_commit": freeze_data["source_commit"], "freeze_signature": freeze_data["freeze_signature"],
        "input_sha256": freeze_data["input_sha256"], "query_labels_opened": True,
        "score_source": "official Classification Train temporal query rows only",
        "test_or_sample_access": False, "fit_ids": list(FIT_IDS), "fit_failures": [],
        "origins": origins, "gate_checks": all_gates, "all_gates_pass": passed,
        "ledger_snapshot_before_completion": snapshot,
        "conditional_e79_reserved": False, "conditional_e79_started": False,
        "interpretation": "Fixed three-channel continuous moments over a frozen unigram space; no causal or public-score claim.",
        "limitations": [
            "E67 and E75 are overlapping adaptive temporal views; W6 overlaps cycles 76-79.",
            "Fit-only label populations are origin-specific; no E67 query metrics were opened until all four predictions passed prelabel QA.",
            "The fixed route is a diagnostic candidate only; E79 and Kaggle require separate Sol review/authorization.",
        ],
    }
    from scripts.verify_optimization_v3_positional_moments_v1 import verify_scored_results
    qa = verify_scored_results(ROOT, RUN_ROOT, freeze_data, core)
    if qa.get("status") != "PASS":
        raise MomentsPilotError(f"Independent scored-result QA failed: {qa}")
    core["independent_verification_sha256"] = _sha_obj(qa)
    _atomic_json(RUN_ROOT / "RESULTS.json", core)
    _atomic_json(RUN_ROOT / "QA.json", qa)
    _atomic_json(REPORT_ROOT / "RESULTS.json", core)
    _atomic_json(REPORT_ROOT / "QA.json", qa)
    _plot_results(core)
    _write_report(core, qa)
    result_sha = sha256_file(RUN_ROOT / "RESULTS.json")
    _append_event(LEDGER, {
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze_data["freeze_signature"], "source_commit": freeze_data["source_commit"],
        "status": core["status"], "result_sha256": result_sha,
        "query_labels_opened": True, "test_or_sample_access": False,
        "conditional_e79_reserved": False, "conditional_e79_started": False,
    }, official=True)
    final_events, final_snapshot = _ledger_snapshot(LEDGER)
    accounting = {
        "status": "COMPLETED_4_PILOT_FITS_NO_E79",
        "completion_event": final_events[-1], "completion_event_sha256": _sha_obj(final_events[-1]),
        "result_sha256": result_sha, "qa_sha256": sha256_file(RUN_ROOT / "QA.json"),
        "ledger_sha256_after_completion": sha256_file(LEDGER), "ledger_snapshot_after_completion": final_snapshot,
        "fit_ids": list(FIT_IDS), "fit_failures": [],
        "conditional_e79_fit_id": f"{CARD_ID}_E79_REAL", "conditional_e79_reserved": False,
        "conditional_e79_started": False, "test_or_sample_access": False,
    }
    _atomic_json(RUN_ROOT / "ACCOUNTING.json", accounting)
    _atomic_json(REPORT_ROOT / "ACCOUNTING.json", accounting)
    _copy_report_outputs_to_run()
    return core


def _copy_report_outputs_to_run() -> None:
    for name in ("RESULTS.json", "QA.json", "ACCOUNTING.json", "RESULTS.md", "primary_gates.png", "per_cycle_delta.png"):
        source = REPORT_ROOT / name
        if source.is_file():
            target = RUN_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def _synthetic_lifecycle(temp_root: Path, prefix_path: Path, *, source_commit: str,
                         source_hashes: Mapping[str, str]) -> dict[str, Any]:
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_ledger = temp_root / "budget.jsonl"
    shutil.copyfile(prefix_path, temp_ledger)
    prefix_sha = sha256_file(temp_ledger)
    prefix_events, prefix_snapshot = _ledger_snapshot(temp_ledger)
    if prefix_snapshot != BASE_SNAPSHOT:
        raise MomentsPilotError(f"Synthetic ledger prefix is not the authorized 137-fit state: {prefix_snapshot}")
    fake_unsigned = {"synthetic": True, "source_commit": source_commit, "prefix_sha256": prefix_sha}
    fake_freeze = {
        "source_commit": source_commit,
        "freeze_signature": _sha_obj(fake_unsigned),
        "source_hashes": dict(source_hashes), "input_sha256": "SYNTHETIC_TRAIN_SHA256",
        "pre_reservation_ledger_sha256": prefix_sha,
        "pre_reservation_ledger_bytes": temp_ledger.stat().st_size,
        "pre_reservation_ledger_rows": len(prefix_events),
    }
    _append_pilot_reservation(temp_ledger, fake_freeze, official=False)
    if offset.ledger_snapshot(offset._ledger_events(temp_ledger)) != NEW_PRE_FIT_SNAPSHOT:
        raise MomentsPilotError("Synthetic temp-ledger reserve did not bind expected fit-count/cap signatures")
    docs_fit = ["alpha beta alpha", "beta gamma", "gamma alpha", "delta beta", "alpha delta", "beta beta gamma", "gamma delta", "alpha gamma"]
    docs_score = ["alpha beta", "gamma delta", "unseen omega", "", "beta alpha gamma", "delta delta",
                  "alpha alpha", "gamma", "unknown beta", "beta gamma", "café alpha", "omega delta"]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.models.classification_models import _TFIDF_PARAMS
    vectorizer = TfidfVectorizer(**{**_TFIDF_PARAMS, "min_df": 1, "max_df": 1.0, "max_features": 50})
    vectorizer.fit(docs_fit)
    x_fit_real, audit_fit_real = build_moment_matrix(docs_fit, vectorizer, shuffled=False)
    x_fit_shuf, audit_fit_shuf = build_moment_matrix(docs_fit, vectorizer, shuffled=True)
    x_score_real, audit_score_real = build_moment_matrix(docs_score, vectorizer, shuffled=False)
    x_score_shuf, audit_score_shuf = build_moment_matrix(docs_score, vectorizer, shuffled=True)
    from scripts.verify_optimization_v3_positional_moments_v1 import _rebuild_moments
    if (_rebuild_moments(docs_fit, vectorizer, shuffled=False).shape != x_fit_real.shape
            or sha256_csr(_rebuild_moments(docs_fit, vectorizer, shuffled=False)) != sha256_csr(x_fit_real)
            or sha256_csr(_rebuild_moments(docs_fit, vectorizer, shuffled=True)) != sha256_csr(x_fit_shuf)
            or sha256_csr(_rebuild_moments(docs_score, vectorizer, shuffled=False)) != sha256_csr(x_score_real)
            or sha256_csr(_rebuild_moments(docs_score, vectorizer, shuffled=True)) != sha256_csr(x_score_shuf)):
        raise MomentsPilotError("Independent synthetic moment reconstruction differs from runner matrices")
    if audit_fit_real["bag_channel_sha256"] != audit_fit_shuf["bag_channel_sha256"] or audit_score_real["token_count_sha256"] != audit_score_shuf["token_count_sha256"]:
        raise MomentsPilotError("Synthetic shuffle did not preserve bag/token length")
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    z0 = np.linspace(-0.25, 0.25, len(y), dtype=np.float64)
    score_z0 = np.linspace(-0.2, 0.2, len(docs_score), dtype=np.float64)
    row_ids = np.asarray([f"SYN-{i:03d}" for i in range(len(y))], dtype=str)
    row_pos = np.arange(len(y), dtype=np.int64)
    call_counter = {fit_id: 0 for fit_id in FIT_IDS}
    results: list[dict[str, Any]] = []
    for i, fit_id in enumerate(FIT_IDS):
        origin = 67 if "E67" in fit_id else 75
        arm = "REAL" if fit_id.endswith("_REAL") else "SHUFFLED"
        x = x_fit_real if arm == "REAL" else x_fit_shuf
        # Count the only place the optimizer is called, for durable-resume proof.
        original_fit = fit_offset_coefficients
        def counted_fit(*args: Any, **kwargs: Any):
            call_counter[fit_id] += 1
            return original_fit(*args, **kwargs)
        globals()["fit_offset_coefficients"] = counted_fit
        try:
            first = _run_one_fit(fit_id, origin=origin, arm=arm, x=x, z0=z0, y=y,
                                 fit_rows=row_pos, fit_ids=row_ids,
                                 freeze_data=fake_freeze, ledger_path=temp_ledger,
                                 run_root=temp_root, official=False,
                                 interrupt_after_durable=(i == 0), resource={"jobs": 1, "threads": 2, "synthetic": True})
        finally:
            globals()["fit_offset_coefficients"] = original_fit
        if i == 0:
            if first.get("status") != "durable_pending_for_synthetic_resume":
                raise MomentsPilotError("Synthetic interrupted save state did not persist artifact and receipt")
            recovered = _resume_durable_fit(fit_id, freeze_data=fake_freeze, ledger_path=temp_ledger,
                                             artifact_path=temp_root / "units" / f"{fit_id}.npz",
                                             receipt_path=temp_root / "receipts" / f"{fit_id}.json", official=False)
            if recovered != "recovered_saved_without_refit" or call_counter[fit_id] != 1:
                raise MomentsPilotError("Synthetic recovery re-fit an already durable optimizer artifact")
            results.append({"fit_id": fit_id, "status": recovered, "optimizer_calls": call_counter[fit_id]})
        else:
            if first.get("status") != "saved" or call_counter[fit_id] != 1:
                raise MomentsPilotError(f"Synthetic fit lifecycle failed for {fit_id}: {first}")
            results.append({"fit_id": fit_id, "status": first["status"], "optimizer_calls": call_counter[fit_id]})
    if sum(item["optimizer_calls"] for item in results) != 4:
        raise MomentsPilotError("Synthetic exact-four optimizer count mismatch")

    present = np.asarray([True, True, False, False, True, True, False, True, True, False, True, True], dtype=bool)
    p_a02 = np.asarray([0.4, 0.6, 0.91, 0.12, 0.33, 0.66, 0.22, 0.7, 0.3, 0.84, 0.61, 0.48], dtype=np.float64)
    p_old = np.asarray([0.45, 0.55, 0.8, 0.2, 0.5, 0.5, 0.6, 0.4, 0.65, 0.35, 0.52, 0.44], dtype=np.float64)
    candidate_outputs: dict[str, np.ndarray] = {}
    probability_outputs: dict[str, np.ndarray] = {}
    for fit_id in FIT_IDS:
        origin = 67 if "E67" in fit_id else 75
        arm = "REAL" if fit_id.endswith("_REAL") else "SHUFFLED"
        gamma = _atomic_load_npz(temp_root / "units" / f"{fit_id}.npz", ["gamma"])["gamma"]
        x_score = x_score_real if arm == "REAL" else x_score_shuf
        q = offset_probability(x_score, score_z0, gamma)
        mix, current = fixed_blends(q, p_a02, p_old, present)
        if not np.array_equal(mix[~present], p_old[~present]) or not np.array_equal(current[~present], p_old[~present]):
            raise MomentsPilotError("Synthetic fixed route changed missing-stream rows")
        candidate_outputs[fit_id] = mix
        probability_outputs[fit_id] = q
    if not all(np.isfinite(value).all() for value in candidate_outputs.values()):
        raise MomentsPilotError("Synthetic prelabel predictions are not finite")
    from scripts.verify_optimization_v3_positional_moments_v1 import verify_synthetic_prelabel
    synthetic_prelabel_checks = []
    for fit_id in FIT_IDS:
        arm = "REAL" if fit_id.endswith("_REAL") else "SHUFFLED"
        matrix = x_score_real if arm == "REAL" else x_score_shuf
        gamma = _atomic_load_npz(temp_root / "units" / f"{fit_id}.npz", ["gamma"])["gamma"]
        _, expected_mix = fixed_blends(probability_outputs[fit_id], p_a02, p_old, present)
        synthetic_prelabel_checks.append(verify_synthetic_prelabel(
            matrix, gamma, score_z0, p_a02, p_old, present,
            probability_outputs[fit_id], candidate_outputs[fit_id], expected_mix))
    # Synthetic metric/gate path exercises the same all-view scoring helper with supported/unsupported groups.
    synthetic_y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    synthetic_bundle = {
        "score_cycles": np.arange(68, 80, dtype=np.int64),
        "present_mask": present, "entry_friction": np.asarray([0, 1] * 6, dtype=np.int8),
        "text_load_group": np.asarray([0, 1, 2, 3] * 3, dtype=np.int8),
        "prob_candidate_real": candidate_outputs[FIT_IDS[0]],
        "prob_candidate_shuffled": candidate_outputs[FIT_IDS[1]],
        "prob_mix50": candidate_outputs[FIT_IDS[0]], "prob_q_real": candidate_outputs[FIT_IDS[0]],
        "prob_q_shuffled": candidate_outputs[FIT_IDS[1]], "prob_pP": p_old,
        "prob_a02": p_a02, "prob_bag": p_old,
    }
    score_probe, gates_probe = _score_origin(67, synthetic_bundle, synthetic_y)
    if set(score_probe["views"]) != {"H12_E67", "W4", "W5", "W6"} or not gates_probe:
        raise MomentsPilotError("Synthetic scoring/gate path did not exercise temporal views")
    wrong_sig_rejected = False
    try:
        _assert_active_freeze("wrong-freeze-signature", fake_freeze)
    except MomentsPilotError:
        wrong_sig_rejected = True
    if not wrong_sig_rejected:
        raise MomentsPilotError("Synthetic callback accepted a wrong active freeze signature")
    final_events, final_snapshot = _ledger_snapshot(temp_ledger)
    if (final_snapshot["fit_starts"] != 137 or final_snapshot["fit_saved"] != 131
            or final_snapshot["fit_pending"] != 0 or len(final_events) != 356):
        raise MomentsPilotError(f"Synthetic temp-ledger lifecycle/accounting differs: {final_snapshot}")
    checked_events, checked_snapshot = _check_live_ledger(fake_freeze, ledger_path=temp_ledger, run_root=temp_root)
    if checked_events != final_events or checked_snapshot != final_snapshot:
        raise MomentsPilotError("Synthetic full-ledger validator differs from its event accounting")
    return {
        "status": "PASS", "source_commit": source_commit, "source_hashes": _source_hashes(),
        "production_ledger_sha256": EXPECTED_LEDGER_SHA,
        "temporary_ledger_initial_sha256": prefix_sha,
        "temporary_ledger_final_snapshot": final_snapshot,
        "temporary_ledger_fit_ids": list(FIT_IDS), "synthetic_optimizer_fits": 4,
        "fit_lifecycle": results, "durable_resume_without_refit": True,
        "wrong_freeze_callback_rejected": wrong_sig_rejected,
        "feature_cases": {"fit_real": audit_fit_real, "fit_shuffled": audit_fit_shuf,
                           "score_real": audit_score_real, "score_shuffled": audit_score_shuf},
        "prelabel_integrity": "PASS", "synthetic_prelabel_checks": synthetic_prelabel_checks,
        "synthetic_scoring_and_gates": "PASS",
        "query_labels_read": False, "test_or_sample_access": False,
    }


def synthetic() -> dict[str, Any]:
    source_hashes = _source_hashes()
    before = sha256_file(LEDGER)
    if before == EXPECTED_LEDGER_SHA:
        prefix_path = LEDGER
    else:
        freeze_path = RUN_ROOT / "ledger_pre_reservation.jsonl"
        if not freeze_path.is_file() or sha256_file(freeze_path) != EXPECTED_LEDGER_SHA:
            raise MomentsPilotError("Synthetic QA after live reservation requires the frozen pre-reservation ledger copy")
        prefix_path = freeze_path
    with tempfile.TemporaryDirectory(prefix="pos_moments_v1_synthetic_") as temp:
        result = _synthetic_lifecycle(Path(temp), prefix_path, source_commit=_git("rev-parse", "HEAD"),
                                      source_hashes=source_hashes)
    after = sha256_file(LEDGER)
    result["production_ledger_unchanged"] = before == after
    result["production_ledger_sha256_before"] = before
    result["production_ledger_sha256_after"] = after
    if not result["production_ledger_unchanged"]:
        raise MomentsPilotError("Synthetic lifecycle modified the production shared ledger")
    _atomic_json(SYNTHETIC_REPORT, result)
    _atomic_json(RUN_ROOT / "SYNTHETIC_QA.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["synthetic", "preflight", "freeze", "reserve", "fit", "predictions", "verify-prelabel", "score"])
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            result = synthetic()
        elif args.command == "preflight":
            result = preflight()
        elif args.command == "freeze":
            result = freeze()
        elif args.command == "reserve":
            result = reserve()
        elif args.command == "fit":
            result = run_fits()
        elif args.command == "predictions":
            result = build_predictions()
        elif args.command == "verify-prelabel":
            result = verify_prelabel()
        else:
            result = score()
    except Exception as exc:
        print(f"POS_MOMENTS_V1 {args.command} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
