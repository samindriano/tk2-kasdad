"""Run the frozen six-fit known-bigram position pilot.

Stages are deliberately separate: preflight -> synthetic -> freeze -> reserve ->
run-all -> predictions -> verify-prelabel -> score. Query labels are first read
by score(), after independent verification of durable prediction bundles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.models.optimization_v3_bigram_position_pilot_v1 import (
    reconstruct_original_pos_logit, route_candidate, zero_gamma_equivalence,
)
from src.models.optimization_v3_bigram_position_readiness_v1 import build_bigram_blocks
from src.models.positional_bag_offset_v1 import fit_offset_coefficients
from scripts import optimization_v3_positional_bag_offset_v1 as offset
from scripts import optimization_v3_positional_contrast_v1 as contrast
from scripts.verify_optimization_v3_bigram_position_pilot_v1 import verify_schema_successor_cache


CARD_ID = "CLS_BIGRAM_POSITION_PILOT_V1"
STAGE = "SIX_FIT_TEMPORAL_PILOT"
AMENDMENT_ID = "CLS_BIGRAM_POSITION_PILOT_V1_CAP149_2026_11_04"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = (
    f"{CARD_ID}_E67_BAG", f"{CARD_ID}_E67_REAL", f"{CARD_ID}_E67_SHUFFLED",
    f"{CARD_ID}_E75_BAG", f"{CARD_ID}_E75_REAL", f"{CARD_ID}_E75_SHUFFLED",
)
ORIGINS = (67, 75)
ARMS = ("BAG", "REAL", "SHUFFLED")
TRAIN = ROOT / "data/raw/classification/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/BIGRAM_POSITION_PILOT_V1"
REPORT_ROOT = ROOT / "reports/optimization_v3/bigram_position_pilot_v1"
ACTIVE_FREEZE = RUN_ROOT / "FREEZE_ACTIVE.json"
PREVIOUS_BAG_ROOT = offset.RUN_ROOT
PREVIOUS_CONTRAST_ROOT = contrast.RUN_ROOT
PREVIOUS_BAG_FREEZE = PREVIOUS_BAG_ROOT / "FREEZE.json"
EXPECTED_TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_PARENT_FREEZE = "ba58b1e446a6be8546789c5a56596fbdbf075efc0ee19fcc1f41e02af7b419d7"
PRE_FIT_SNAPSHOT = {
    "event_rows": 361, "global_cap": 143,
    "effective_category_caps": {"continuation": 52, "control_replay": 2,
                                 "pilot": 88, "technical_retry": 1},
    "active_reservations_by_category": {"continuation": 50, "control_replay": 2,
                                        "pilot": 85, "technical_retry": 1},
    "active_reservations": 138, "fit_starts": 138, "fit_saved": 132,
    "fit_failed": 6, "fit_pending": 0,
}
POST_RESERVE_SNAPSHOT = {
    "event_rows": 363, "global_cap": 149,
    "effective_category_caps": {"continuation": 52, "control_replay": 2,
                                 "pilot": 94, "technical_retry": 1},
    "active_reservations_by_category": {"continuation": 50, "control_replay": 2,
                                        "pilot": 91, "technical_retry": 1},
    "active_reservations": 144, "fit_starts": 138, "fit_saved": 132,
    "fit_failed": 6, "fit_pending": 0,
}
POST_RESERVE_CAPS = {"continuation": 52, "control_replay": 2, "pilot": 94,
                     "technical_retry": 1}
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3

SOURCE_PATHS = (
    "scripts/optimization_v3_bigram_position_pilot_v1.py",
    "scripts/verify_optimization_v3_bigram_position_pilot_v1.py",
    "src/models/optimization_v3_bigram_position_pilot_v1.py",
    "src/models/optimization_v3_bigram_position_readiness_v1.py",
    "scripts/optimization_v3_bigram_position_readiness_v1.py",
    "src/models/positional_bag_offset_v1.py",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/optimization_v3_positional_contrast_v1.py",
    "src/models/positional_text_features.py",
    "tests/test_optimization_v3_bigram_position_pilot_v1.py",
    "reports/optimization_v3/bigram_position_pilot_v1/RECIPE.json",
    "reports/optimization_v3/bigram_position_pilot_v1/PROPOSED_CARD.md",
    "reports/optimization_v3/bigram_position_pilot_v1/PREFIT_SCHEMA_REPAIR.md",
    "reports/optimization_v3/bigram_position_readiness_v1/SOL_PILOT_AUTHORIZATION.md",
)


class PilotError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    return offset.sha256_file(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    offset._atomic_json(path, dict(payload))


def _source_hashes() -> dict[str, str]:
    out = {}
    for name in SOURCE_PATHS:
        path = ROOT / name
        if not path.is_file():
            raise PilotError(f"Required source is missing: {name}")
        out[name] = _sha_file(path)
    return out


def _git(*args: str) -> str:
    return offset._git(*args)


def _current_ledger() -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    events = offset._ledger_events(LEDGER)
    return events, offset.ledger_snapshot(events), _sha_file(LEDGER)


def _load_frozen_vectorizer(origin: int, input_sha: str):
    unit = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_C2B_TFIDF"
    artifact = offset.PILOT_ROOT / "units" / f"{unit}.joblib"
    receipt_path = offset.PILOT_ROOT / "receipts" / f"{unit}.json"
    receipt = _json(receipt_path)
    if _sha_file(artifact) != receipt.get("artifact_sha256") or receipt.get("input_sha256") != input_sha:
        raise PilotError(f"E{origin} frozen vectorizer receipt/hash mismatch")
    import joblib
    vectorizer = joblib.load(artifact)
    params = vectorizer.get_params(deep=False)
    expected = {"analyzer": "word", "ngram_range": (1, 2), "strip_accents": "unicode",
                "lowercase": True, "token_pattern": r"(?u)\b\w\w+\b"}
    for name, value in expected.items():
        actual = tuple(params[name]) if name == "ngram_range" else params[name]
        if actual != value:
            raise PilotError(f"E{origin} frozen vectorizer parameter changed: {name}")
    return vectorizer, {"path": artifact.relative_to(ROOT).as_posix(),
                        "artifact_sha256": _sha_file(artifact),
                        "receipt_sha256": _sha_file(receipt_path),
                        "fit_id": unit, "fit_receipt": receipt}


def _load_anchor(origin: int, identity: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bundle_path = PREVIOUS_CONTRAST_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    ready = _json(PREVIOUS_CONTRAST_ROOT / "PREDICTIONS_READY.json")
    entry = ready.get("predictions", {}).get(f"E{origin}", {})
    if ready.get("query_labels_opened") is not False or _sha_file(bundle_path) != entry.get("sha256"):
        raise PilotError(f"Frozen POS_CONTRAST E{origin} output is not ready/hash-valid")
    keys = ["fit_rows", "score_rows", "score_record_ids", "score_cycles", "present_mask",
            "prob_q_real", "prob_pP", "prob_a02", "prob_mix50"]
    values = contrast._load_named_npz(bundle_path, keys)
    a02_meta, a02 = contrast._load_a02(origin, include_groups=True)
    if not (np.array_equal(values["score_rows"], a02["score_rows"])
            and np.array_equal(values["score_record_ids"].astype(str), a02["score_record_ids"].astype(str))
            and np.array_equal(values["score_cycles"], a02["score_cycles"])):
        raise PilotError(f"E{origin} A02 and frozen POS prediction rows differ")
    rows = np.asarray(values["score_rows"], dtype=np.int64)
    sub = identity.iloc[rows]
    if (not np.array_equal(sub["record_id"].astype(str).to_numpy(), values["score_record_ids"].astype(str))
            or not np.array_equal(sub["cycle_code"].to_numpy(dtype=np.int64), values["score_cycles"].astype(np.int64))):
        raise PilotError(f"E{origin} frozen query row IDs/cycles do not match Train identity")
    present = np.asarray(values["present_mask"], dtype=bool)
    if not np.array_equal(np.asarray(a02["missing_mask"], dtype=bool), ~present):
        raise PilotError(f"E{origin} A02 availability mask differs from frozen POS mask")
    if not np.array_equal(np.asarray(values["prob_mix50"])[~present],
                          np.asarray(values["prob_pP"])[~present]):
        raise PilotError(f"E{origin} frozen MIX50 missing fallback is not exact pP")
    expected_mix = 0.5 * np.asarray(values["prob_pP"], dtype=np.float64) + 0.5 * np.asarray(values["prob_a02"], dtype=np.float64)
    expected_mix[~present] = values["prob_pP"][~present]
    if not np.array_equal(expected_mix, values["prob_mix50"]):
        raise PilotError(f"E{origin} cached MIX50 differs from frozen policy formula")
    return ({"bundle_sha256": _sha_file(bundle_path), "ready_sha256": _sha_file(PREVIOUS_CONTRAST_ROOT / "PREDICTIONS_READY.json"),
             "a02_prediction_json_sha256": a02_meta["prediction_json_sha256"],
             "a02_receipt_sha256": a02_meta["receipt_sha256"],
             "a02_safe_array_hashes": a02_meta["safe_array_hashes"]},
            {**values, "entry_friction": np.asarray(a02["entry_friction"]).astype(str),
             "text_load_group": np.asarray(a02["text_load_group"], dtype=np.int8)})


def _load_original_pos_gamma(origin: int) -> tuple[np.ndarray, dict[str, Any]]:
    fit_id = f"POS_BAG_OFFSET_V1_E{origin}_REAL"
    receipt_path = PREVIOUS_BAG_ROOT / "receipts" / f"{fit_id}.json"
    receipt = _json(receipt_path)
    artifact = ROOT / receipt["artifact_path"]
    if (_sha_file(artifact) != receipt.get("artifact_sha256")
            or receipt.get("freeze_signature") != EXPECTED_PARENT_FREEZE
            or receipt.get("fit_id") != fit_id):
        raise PilotError(f"Original POS gamma artifact/receipt mismatch for E{origin}")
    with np.load(artifact, allow_pickle=False) as values:
        gamma = np.asarray(values["gamma"], dtype=np.float64).copy()
    if offset.sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise PilotError(f"Original E{origin} gamma logical hash mismatch")
    return gamma, {"fit_id": fit_id, "artifact_sha256": _sha_file(artifact),
                   "receipt_sha256": _sha_file(receipt_path), "gamma_sha256": offset.sha256_array(gamma),
                   "source_commit": receipt.get("source_commit"), "freeze_signature": receipt.get("freeze_signature")}


def _parent_payload(origin: int, identity: pd.DataFrame):
    parent_freeze = _json(PREVIOUS_BAG_FREEZE)
    if parent_freeze.get("freeze_signature") != EXPECTED_PARENT_FREEZE:
        raise PilotError("Frozen POS_BAG_OFFSET parent signature changed")
    manifest, prediction, bag, payload = offset._load_origin_payload(origin, parent_freeze)
    fit_rows = np.asarray(prediction["fit_rows"], dtype=np.int64)
    score_rows = np.asarray(prediction["score_rows"], dtype=np.int64)
    if len(fit_rows) != (29705 if origin == 67 else 42117) or len(score_rows) != (20903 if origin == 67 else 8491):
        raise PilotError(f"E{origin} parent row populations differ from frozen temporal plan")
    fr = identity.iloc[fit_rows]
    qr = identity.iloc[score_rows]
    if (fr["cycle_code"].max() != origin or fr["cycle_code"].min() != 47
            or qr["cycle_code"].min() != origin + 1 or qr["cycle_code"].max() != 79):
        raise PilotError(f"E{origin} parent rows do not follow cutoff boundaries")
    matrices = payload["matrices"]
    base_fit = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
    base_query = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
    augmented_fit = sparse.csr_matrix(matrices["fit_position"], dtype=np.float64)
    augmented_query = sparse.csr_matrix(matrices["score_position"], dtype=np.float64)
    # Sparse logical hashes avoid materializing a multi-gigabyte dense array.
    if offset.sha256_csr(augmented_fit[:, :base_fit.shape[1]]) != offset.sha256_csr(base_fit):
        raise PilotError(f"E{origin} fit base prefix differs inside original position matrix")
    if offset.sha256_csr(augmented_query[:, :base_query.shape[1]]) != offset.sha256_csr(base_query):
        raise PilotError(f"E{origin} query base prefix differs inside original position matrix")
    pos_fit = augmented_fit[:, base_fit.shape[1]:].tocsr()
    pos_query = augmented_query[:, base_query.shape[1]:].tocsr()
    gamma_old, gamma_record = _load_original_pos_gamma(origin)
    if len(gamma_old) != pos_fit.shape[1] or pos_query.shape[1] != pos_fit.shape[1]:
        raise PilotError(f"E{origin} old unigram-position gamma width differs from frozen matrix")
    z_fit, z_fit_plus_pos = reconstruct_original_pos_logit(base_fit, bag.coef_[0], bag.intercept_, pos_fit, gamma_old)
    del z_fit
    z_query_bag, z_query = reconstruct_original_pos_logit(base_query, bag.coef_[0], bag.intercept_, pos_query, gamma_old)
    return {
        "manifest": manifest, "prediction": prediction, "bag": bag, "payload": payload,
        "fit_rows": fit_rows, "score_rows": score_rows,
        "base_fit": base_fit, "base_query": base_query,
        "pos_fit": pos_fit, "pos_query": pos_query,
        "z_fit": z_fit_plus_pos, "z_query": z_query,
        "z_query_bag": z_query_bag, "gamma_record": gamma_record,
    }


def _atomic_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    sparse.save_npz(temp, sparse.csr_matrix(matrix, dtype=np.float64))
    os.replace(temp, path)


def _read_fit_labels_only(train_path: Path, fit_rows: np.ndarray) -> np.ndarray:
    """Skip non-fit CSV records before parsing the target column."""
    selected = set(np.asarray(fit_rows, dtype=np.int64).tolist())
    target = pd.read_csv(
        train_path, usecols=["artifact_signal"],
        skiprows=lambda line_index: line_index > 0 and (line_index - 1) not in selected,
    )["artifact_signal"]
    if len(target) != len(selected):
        raise PilotError("Fit-only CSV selection returned an unexpected number of target rows")
    values = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(values, (0, 1)).all():
        raise PilotError("Fit-only labels are not binary 0/1")
    return values


def _cache_origin(origin: int, documents: list[str], identity: pd.DataFrame,
                  raw_present: np.ndarray, train_sha: str) -> dict[str, Any]:
    vectorizer, vector_record = _load_frozen_vectorizer(origin, train_sha)
    parent = _parent_payload(origin, identity)
    parent_info, anchor = _load_anchor(origin, identity)
    rows_fit = parent["fit_rows"]
    rows_query = parent["score_rows"]
    blocks, metadata = build_bigram_blocks(documents, vectorizer)
    if not np.array_equal(raw_present[rows_query], anchor["present_mask"].astype(bool)):
        raise PilotError(f"E{origin} raw text availability differs from frozen route")
    q_replayed = expit(parent["z_query"])
    present_query = anchor["present_mask"].astype(bool)
    max_pos_diff = float(np.max(np.abs(q_replayed[present_query] - anchor["prob_pP"][present_query]), initial=0.0))
    if max_pos_diff > 1e-12:
        raise PilotError(f"E{origin} reconstructed original positional logit does not replay saved probability")
    z0 = zero_gamma_equivalence(
        blocks["real"][rows_query], parent["z_query"], present_mask=present_query,
        p_a02=anchor["prob_a02"], p_p=anchor["prob_pP"], mix50_original=anchor["prob_mix50"],
    )
    if not z0["pass"]:
        raise PilotError(f"E{origin} zero-gamma MIX50 equivalence failed: {z0}")
    cache_dir = RUN_ROOT / f"E{origin}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_hashes = {}
    for arm in ARMS:
        name = arm.lower()
        fit = blocks[name][rows_fit].tocsr()
        query = blocks[name][rows_query].tocsr()
        _atomic_sparse(cache_dir / f"fit_{name}.npz", fit)
        _atomic_sparse(cache_dir / f"query_{name}.npz", query)
        matrix_hashes[f"fit_{name}"] = {"file_sha256": _sha_file(cache_dir / f"fit_{name}.npz"),
                                          "logical_sha256": offset.sha256_csr(fit), "shape": list(fit.shape)}
        matrix_hashes[f"query_{name}"] = {"file_sha256": _sha_file(cache_dir / f"query_{name}.npz"),
                                            "logical_sha256": offset.sha256_csr(query), "shape": list(query.shape)}
    fit_ids = identity.iloc[rows_fit]["record_id"].astype(str).to_numpy(dtype=str)
    score_ids = identity.iloc[rows_query]["record_id"].astype(str).to_numpy(dtype=str)
    score_cycles = identity.iloc[rows_query]["cycle_code"].to_numpy(dtype=np.int64)
    metadata_npz = cache_dir / "frozen_rows.npz"
    offset._atomic_npz(
        metadata_npz, fit_rows=rows_fit, fit_record_ids=fit_ids,
        score_rows=rows_query, score_record_ids=score_ids, score_cycles=score_cycles,
        present_mask=present_query, no_known_bigram=(metadata["known_occurrence_count"][rows_query] == 0),
        z_fit_original=parent["z_fit"], z_query_original=parent["z_query"],
        p_pos_original=q_replayed, p_p=anchor["prob_pP"], p_a02=anchor["prob_a02"],
        mix50=anchor["prob_mix50"], entry_friction=anchor["entry_friction"],
        text_load_group=anchor["text_load_group"], known_fit=metadata["known_occurrence_count"][rows_fit],
        known_query=metadata["known_occurrence_count"][rows_query],
    )
    record = {
        "origin": origin, "vectorizer": vector_record, "parent_manifest_sha256": _sha_file(offset.PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"),
        "parent_prediction_sha256": parent_info["bundle_sha256"],
        "parent_prelabel_verified": True,
        "parent_pos_gamma": parent["gamma_record"],
        "anchor_bundle": parent_info,
        "matrix_hashes": matrix_hashes,
        "row_arrays_sha256": {"fit_rows": offset.sha256_array(rows_fit), "fit_record_ids": offset.sha256_array(fit_ids),
                              "score_rows": offset.sha256_array(rows_query), "score_record_ids": offset.sha256_array(score_ids),
                              "score_cycles": offset.sha256_array(score_cycles), "present_mask": offset.sha256_array(present_query),
                              "metadata_file": _sha_file(metadata_npz)},
        "fit_rows": int(len(rows_fit)), "query_rows": int(len(rows_query)),
        "fit_cycle_range": [47, origin], "query_cycle_range": [origin + 1, 79],
        "feature_widths": {arm: int(blocks[arm.lower()].shape[1]) for arm in ARMS},
        "feature_known_fit_rows": int(np.count_nonzero(metadata["known_occurrence_count"][rows_fit])),
        "feature_known_query_rows": int(np.count_nonzero(metadata["known_occurrence_count"][rows_query])),
        "pair_shuffle_preservation_pass": bool(metadata["pair_bag_preserved"].all() and metadata["segment_sizes_preserved"].all()),
        "original_logit_query_replay_max_abs": max_pos_diff,
        "zero_gamma_equivalence": z0,
        "old_position_fit_block_sha256": offset.sha256_csr(parent["pos_fit"]),
        "old_position_query_block_sha256": offset.sha256_csr(parent["pos_query"]),
        "original_z_fit_sha256": offset.sha256_array(parent["z_fit"]),
        "original_z_query_sha256": offset.sha256_array(parent["z_query"]),
        "fit_labels_read": False, "query_labels_read": False, "test_opened": False,
    }
    del blocks, parent
    return record


def _preserve_current_successor_cache(old_freeze: Mapping[str, Any],
                                      identity: pd.DataFrame) -> dict[str, Any]:
    """Archive migrated cache bytes; never describe them as recovered original bytes."""
    preserved = {}
    for origin in ORIGINS:
        suffix = f"E{origin}"
        source_dir = RUN_ROOT / suffix
        target_dir = RUN_ROOT / "PREFIT_SCHEMA_V1_CURRENT_ARCHIVE" / suffix
        target_dir.mkdir(parents=True, exist_ok=True)
        expected = old_freeze["origins"][suffix]
        names = ["frozen_rows.npz"] + [f"{split}_{arm.lower()}.npz"
                                      for split in ("fit", "query") for arm in ARMS]
        hashes = {}
        for name in names:
            source = source_dir / name
            if not source.is_file():
                raise PilotError(f"Current prefit cache needed for preservation is missing: {source}")
            if name == "frozen_rows.npz":
                expected_sha = expected["row_arrays_sha256"]["metadata_file"]
            else:
                item = expected["matrix_hashes"][name.removesuffix(".npz")]
                expected_sha = item["file_sha256"]
            actual_sha = _sha_file(source)
            target = target_dir / name
            if target.exists():
                if _sha_file(target) != actual_sha:
                    raise PilotError(f"Current migrated cache archive has conflicting bytes: {target}")
            else:
                shutil.copy2(source, target)
            if _sha_file(target) != actual_sha:
                raise PilotError(f"Current migrated cache archive verification failed: {target}")
            hashes[name] = {
                "path": target.relative_to(ROOT).as_posix(),
                "current_source_sha256": actual_sha,
                "archived_sha256": _sha_file(target),
                "old_freeze_expected_sha256": expected_sha,
                "current_bytes_match_old_freeze": actual_sha == expected_sha,
            }
        with np.load(target_dir / "frozen_rows.npz", allow_pickle=False) as rows:
            fit_rows = np.asarray(rows["fit_rows"], dtype=np.int64)
            score_rows = np.asarray(rows["score_rows"], dtype=np.int64)
            expected_rows = expected["row_arrays_sha256"]
            fit_ids = rows["fit_record_ids"].astype(str)
            score_ids = rows["score_record_ids"].astype(str)
            identity_fit_ids = identity.iloc[fit_rows]["record_id"].astype(str).to_numpy(dtype=str)
            identity_score_ids = identity.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
            identity_cycles = identity.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
            checks = {
                "fit_rows": offset.sha256_array(fit_rows) == expected_rows["fit_rows"],
                "fit_record_ids_original_object_hash": offset.sha256_array(fit_ids.astype(object)) == expected_rows["fit_record_ids"],
                "score_rows": offset.sha256_array(score_rows) == expected_rows["score_rows"],
                "score_record_ids_original_object_hash": offset.sha256_array(score_ids.astype(object)) == expected_rows["score_record_ids"],
                "score_cycles": offset.sha256_array(identity_cycles) == expected_rows["score_cycles"],
                "identity_alignment": np.array_equal(fit_ids, identity_fit_ids)
                and np.array_equal(score_ids, identity_score_ids)
                and np.array_equal(rows["score_cycles"], identity_cycles),
                "present_mask": offset.sha256_array(rows["present_mask"]) == expected_rows["present_mask"],
                "z_fit_original": offset.sha256_array(rows["z_fit_original"]) == expected["original_z_fit_sha256"],
                "z_query_original": offset.sha256_array(rows["z_query_original"]) == expected["original_z_query_sha256"],
            }
        if not all(checks.values()):
            raise PilotError(f"Current migrated E{origin} row cache fails old-freeze/official-identity checks: {checks}")
        if any(not hashes[name]["current_bytes_match_old_freeze"] for name in names if name != "frozen_rows.npz"):
            raise PilotError(f"Current migrated E{origin} matrix bytes differ from old freeze")
        hashes["row_array_checks_against_old_freeze"] = checks
        hashes["old_row_metadata_bytes_recovered"] = False
        preserved[suffix] = hashes
    return preserved


def preflight() -> dict[str, Any]:
    if not TRAIN.is_file() or not LEDGER.is_file():
        raise PilotError("Official Classification Train or live Research Next ledger is missing")
    train_sha = _sha_file(TRAIN)
    if train_sha != EXPECTED_TRAIN_SHA:
        raise PilotError("Official Train input hash differs from the frozen dataset")
    # This parent verifier reads identity and named non-target artifact arrays only.
    parent_audit = offset.verify_saved_inputs(include_current_code=False)
    old_freeze = _json(PREVIOUS_BAG_FREEZE)
    parent_prelabel = offset._read_json(PREVIOUS_BAG_ROOT / "PRELABEL_VERIFICATION.json")
    if old_freeze.get("freeze_signature") != EXPECTED_PARENT_FREEZE or parent_prelabel.get("status") != "PASS":
        raise PilotError("Frozen POS_BAG parent freeze/prelabel verification is not valid")
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    identity = offset._read_identity_frame()
    if (len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any()
            or not frame["record_id"].astype(str).equals(identity["record_id"].astype(str))):
        raise PilotError("Train identity contract failed")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise PilotError("Train cycle coverage is incomplete or unexpected")
    # Feature values are permitted in target-free preflight; target is excluded by construction.
    text = pd.read_csv(TRAIN, usecols=["signal_stream"])["signal_stream"]
    documents = ["" if pd.isna(value) else str(value) for value in text.tolist()]
    present = np.asarray([bool(x.strip()) for x in documents], dtype=bool)
    del text
    embedded_linebreak_rows = int(sum(("\r" in value) or ("\n" in value) for value in documents))
    if embedded_linebreak_rows:
        blocked = {"schema_version": 1, "card_id": CARD_ID,
                   "status": "PREFIT_BLOCKED_EMBEDDED_LINEBREAKS",
                   "input_sha256": train_sha, "input_rows": len(documents),
                   "fit_label_reader": {"method": "pandas read_csv skiprows over non-prefix physical records, after FIT_STARTED",
                                        "embedded_CR_or_LF_rows": embedded_linebreak_rows,
                                        "unsafe_linebreaks_found": True},
                   "target_columns_read": [], "labels_opened": False,
                   "fit_started": False, "query_labels_read": False,
                   "test_opened": False}
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        _write_json(REPORT_ROOT / "PREFLIGHT_BLOCKER.json", blocked)
        raise PilotError(f"Fit-only CSV skiprows is unsafe: {embedded_linebreak_rows} signal_stream rows contain CR/LF")
    events, snapshot, ledger_sha = _current_ledger()
    reservation_events = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                          and e.get("reservation_key") == RESERVATION_KEY]
    own_starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    parent_freeze_signature = None
    preserved_initial_cache = None
    prior_freeze = None
    if snapshot == PRE_FIT_SNAPSHOT:
        if reservation_events:
            raise PilotError("Six-fit reservation exists while ledger still reports pre-reservation caps")
        preflight_mode = "INITIAL_PREFIT"
        output_path = REPORT_ROOT / "PREFLIGHT.json"
    elif snapshot == POST_RESERVE_SNAPSHOT:
        prior = _json(RUN_ROOT / "FREEZE.json")
        if (len(reservation_events) != 1 or reservation_events[0].get("fit_ids") != list(FIT_IDS)
                or reservation_events[0].get("freeze_signature") != prior.get("freeze_signature")
                or reservation_events[0].get("source_commit") != prior.get("source_commit")
                or own_starts):
            raise PilotError("Post-reservation prefit repair requires the unique untouched reservation and zero starts")
        preflight_mode = "PREFIT_SCHEMA_SUCCESSOR_NO_FITS_STARTED"
        parent_freeze_signature = prior["freeze_signature"]
        prior_freeze = prior
        preserved_initial_cache = _preserve_current_successor_cache(prior, identity)
        output_path = REPORT_ROOT / "PREFLIGHT_SUCCESSOR.json"
    else:
        raise PilotError(f"Live ledger differs from initial or reserved untouched state: {snapshot}")
    try:
        source_commit = _git("rev-parse", "HEAD")
    except Exception as exc:
        raise PilotError(f"Cannot identify source commit: {exc}") from exc
    origins = {}
    for origin in ORIGINS:
        origins[f"E{origin}"] = _cache_origin(origin, documents, identity, present, train_sha)
    cache_migration = None
    if prior_freeze is not None:
        cache_migration = verify_schema_successor_cache(
            ROOT, RUN_ROOT, RUN_ROOT / "PREFIT_SCHEMA_V1_CURRENT_ARCHIVE", prior_freeze)
        if cache_migration.get("status") != "PASS":
            raise PilotError(f"Independent target-free successor-cache verification failed: {cache_migration.get('failures')}")
        for origin in ORIGINS:
            suffix = f"E{origin}"
            rebuilt = origins[suffix]
            old = prior_freeze["origins"][suffix]
            if (rebuilt["anchor_bundle"] != old["anchor_bundle"]
                    or rebuilt["parent_pos_gamma"] != old["parent_pos_gamma"]
                    or rebuilt["parent_manifest_sha256"] != old["parent_manifest_sha256"]
                    or rebuilt["parent_prediction_sha256"] != old["parent_prediction_sha256"]
                    or rebuilt["original_z_fit_sha256"] != old["original_z_fit_sha256"]
                    or rebuilt["original_z_query_sha256"] != old["original_z_query_sha256"]):
                raise PilotError(f"Rebuilt E{origin} parent anchor/logit lineage differs from original freeze")
            for name, matrix in old["matrix_hashes"].items():
                if (rebuilt["matrix_hashes"][name]["file_sha256"] != matrix["file_sha256"]
                        or rebuilt["matrix_hashes"][name]["logical_sha256"] != matrix["logical_sha256"]):
                    raise PilotError(f"Rebuilt E{origin} {name} matrix differs from original freeze")
    events_after, snapshot_after, ledger_sha_after = _current_ledger()
    if snapshot_after != snapshot or ledger_sha_after != ledger_sha:
        raise PilotError("Live ledger changed during target-free preflight/cache reconstruction")
    if any(e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS for e in events_after):
        raise PilotError("An authorized fit started concurrently during target-free preflight")
    result = {
        "schema_version": 1, "card_id": CARD_ID, "status": "PREFIT_PASS",
        "preflight_mode": preflight_mode, "parent_freeze_signature": parent_freeze_signature,
        "source_commit": source_commit, "input_sha256": train_sha, "input_rows": 50608,
        "parent_audit": parent_audit, "parent_prelabel": parent_prelabel,
        "preserved_initial_cache": preserved_initial_cache,
        "cache_migration_verification": cache_migration,
        "live_ledger_sha256": ledger_sha, "live_ledger_snapshot": snapshot,
        "origins": origins,
        "fit_label_reader": {"method": "pandas read_csv skiprows over non-prefix physical records, after FIT_STARTED",
                             "embedded_CR_or_LF_rows": embedded_linebreak_rows,
                             "unsafe_linebreaks_found": False},
        "labels_opened": False, "fit_started": False, "test_opened": False,
        "sample_submission_opened": False, "kaggle_accessed": False,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    return result


def run_synthetic_lifecycle(temp_root: Path) -> dict[str, Any]:
    temp_root = Path(temp_root).resolve()
    if not temp_root.is_relative_to(Path(tempfile.gettempdir()).resolve()) or temp_root == RUN_ROOT.resolve():
        raise PilotError("Synthetic lifecycle root must be an isolated system temp directory")
    production_before = _sha_file(LEDGER)
    temp_root.mkdir(parents=True, exist_ok=True)
    ledger = temp_root / "budget.jsonl"
    gamma_dir = temp_root / "fits"
    score_ids = np.asarray(["é-sample", "二-例", "S3", "S4", "S5", "S6"], dtype="<U16")
    rng = np.random.default_rng(20261104)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)
    z = np.linspace(-0.25, 0.25, len(y), dtype=np.float64)
    xs = {
        "BAG": sparse.csr_matrix(rng.integers(0, 3, size=(len(y), 4)).astype(np.float64)),
        "REAL": sparse.csr_matrix(rng.integers(0, 2, size=(len(y), 12)).astype(np.float64)),
        "SHUFFLED": sparse.csr_matrix(rng.integers(0, 2, size=(len(y), 12)).astype(np.float64)),
    }
    if offset.sha256_array(score_ids) != offset.sha256_array(np.asarray(score_ids, dtype="<U16")):
        raise PilotError("Synthetic Unicode ID hashing is unstable")
    # Exercise a temporary six-fit reservation and the exact production fit primitive.
    offset._append_ledger({"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 0,
                           "category_caps": {"pilot": 0, "continuation": 0,
                                             "control_replay": 0, "technical_retry": 0},
                           "batch_id": "SYNTHETIC_ONLY"}, path=ledger)
    offset._append_ledger({"event": "BUDGET_CAP_AMENDMENT", "global_cap": 6,
                           "effective_caps": {"pilot": 6, "continuation": 0,
                                              "control_replay": 0, "technical_retry": 0},
                           "card_id": CARD_ID, "stage": "SYNTHETIC"}, path=ledger)
    offset._append_ledger({"event": "CARD_FITS_RESERVED", "reservation_key": "SYNTHETIC:" + CARD_ID,
                           "card_id": CARD_ID, "stage": "SYNTHETIC", "category": "pilot", "fits": 6}, path=ledger)
    results = []
    for fit_id in FIT_IDS:
        origin = 67 if "E67" in fit_id else 75
        arm = fit_id.rsplit("_", 1)[1]
        fit_path = gamma_dir / f"{fit_id}.npz"
        receipt_path = gamma_dir / f"{fit_id}.json"
        offset._append_ledger({"event": "FIT_STARTED", "fit_id": fit_id,
                               "reservation_key": "SYNTHETIC:" + CARD_ID,
                               "card_id": CARD_ID, "stage": "SYNTHETIC"}, path=ledger)
        gamma, optimizer = fit_offset_coefficients(xs[arm], z, y, c=1.0)
        offset._atomic_npz(fit_path, gamma=gamma)
        receipt = {"fit_id": fit_id, "origin": origin, "arm": arm,
                   "freeze_signature": "synthetic-freeze",
                   "artifact_sha256": _sha_file(fit_path), "gamma_sha256": offset.sha256_array(gamma),
                   "optimizer": optimizer, "fit_ids_sha256": offset.sha256_array(score_ids[:len(y)]),
                   "test_query_labels_read": False}
        offset._atomic_json(receipt_path, receipt)
        # Simulate the durable-result-before-terminal interruption on E67_BAG.
        if fit_id == FIT_IDS[0]:
            recovered = _recover_one(fit_id, ledger_path=ledger, official=False,
                                     fit_path=fit_path, receipt_path=receipt_path,
                                     expected_freeze="synthetic-freeze")
            if recovered != "resumed_from_durable_receipt":
                raise PilotError("Synthetic durable resume did not recover the saved receipt")
        else:
            offset._append_ledger({"event": "FIT_SAVED", "fit_id": fit_id,
                                   "reservation_key": "SYNTHETIC:" + CARD_ID,
                                   "card_id": CARD_ID, "stage": "SYNTHETIC",
                                   "artifact_sha256": receipt["artifact_sha256"]}, path=ledger)
        results.append({"fit_id": fit_id, "optimizer_success": optimizer["success"],
                        "gamma_sha256": receipt["gamma_sha256"]})
    events = offset._ledger_events(ledger)
    snap = offset.ledger_snapshot(events)
    if snap["fit_starts"] != 6 or snap["fit_saved"] != 6 or snap["fit_failed"] or snap["fit_pending"]:
        raise PilotError(f"Synthetic six-fit ledger lifecycle did not reconcile: {snap}")
    # Persist/reload prediction path with Unicode IDs and missing/no-known routes.
    q_ids = np.asarray(["é-sample", "二-例", "q3", "q4"], dtype="<U16")
    present = np.asarray([True, False, True, True])
    no_known = np.asarray([False, False, True, False])
    zq = np.asarray([-0.1, 0.2, 0.05, -0.3])
    p_a02 = np.asarray([0.7, 0.5, 0.4, 0.2])
    p_p = np.asarray([0.1, 0.3, 0.4, 0.8])
    mix = 0.5 * expit(zq) + 0.5 * p_a02
    mix[~present] = p_p[~present]
    qmat = sparse.csr_matrix([[1.0], [0.0], [0.0], [1.0]])
    gamma = np.asarray([0.05])
    _, routed = route_candidate(qmat, zq, gamma, present_mask=present,
                                no_known_bigram=no_known, p_pos_original=expit(zq),
                                p_a02=p_a02, p_p=p_p, mix50_original=mix)
    out = temp_root / "prediction.npz"
    offset._atomic_npz(out, ids=q_ids, probability=routed, present_mask=present,
                       no_known_bigram=no_known, mix50=mix, p_p=p_p)
    with np.load(out, allow_pickle=False) as loaded:
        if not np.array_equal(loaded["ids"].astype(str), q_ids.astype(str)):
            raise PilotError("Synthetic Unicode record IDs did not survive NPZ persistence")
        got = np.asarray(loaded["probability"], dtype=np.float64)
        if not np.array_equal(got[~present], p_p[~present]) or not np.array_equal(got[present & no_known], mix[present & no_known]):
            raise PilotError("Synthetic missing/no-known routes were not exact after reload")
    synthetic_metric = _metric(np.asarray([0, 1, 1, 0], dtype=np.int8), got)
    synthetic_gates = [
        _gate("synthetic_minimum_boundary", 0.0005, minimum=0.0005),
        _gate("synthetic_minimum_reject", 0.0004, minimum=0.0005),
        _gate("synthetic_maximum_boundary", 0.003, maximum=0.003),
        _gate("synthetic_maximum_reject", 0.0031, maximum=0.003),
    ]
    if (synthetic_metric["n"] != 4 or [x["pass"] for x in synthetic_gates] != [True, False, True, False]):
        raise PilotError("Synthetic metric/gate calculation path failed")
    after = _sha_file(LEDGER)
    if production_before != after:
        raise PilotError("Synthetic lifecycle changed production ledger")
    return {"status": "PASS", "temporary_root": str(temp_root), "fit_count": 6,
            "fit_ids": list(FIT_IDS), "fit_records": results, "resume": "durable receipt -> FIT_SAVED without refit",
            "ledger_snapshot": snap, "unicode_ids_persisted": True,
            "missing_and_no_known_route_exact": True,
            "synthetic_metric": synthetic_metric, "synthetic_gate_paths": synthetic_gates,
            "production_ledger_sha256_before": production_before, "production_ledger_sha256_after": after,
            "query_labels_read": False}


def _recover_one(fit_id: str, *, ledger_path: Path, official: bool,
                 fit_path: Path, receipt_path: Path, expected_freeze: str) -> str:
    events = offset._ledger_events(ledger_path)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) != 1 or terminals:
        raise PilotError(f"Cannot resume fit {fit_id}: start/terminal event state is ambiguous")
    failure_reason = None
    if fit_path.is_file() and receipt_path.is_file():
        try:
            receipt = _json(receipt_path)
            if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != expected_freeze
                    or receipt.get("artifact_sha256") != _sha_file(fit_path)):
                raise PilotError("durable artifact/receipt binding mismatch")
            offset._append_ledger({"event": "FIT_SAVED", "fit_id": fit_id,
                                   "reservation_key": starts[0].get("reservation_key"),
                                   "card_id": CARD_ID, "stage": STAGE,
                                   "artifact_sha256": receipt["artifact_sha256"],
                                   "receipt_sha256": _sha_file(receipt_path),
                                   "resumed_from_durable_receipt": True}, path=ledger_path, official=official)
            return "resumed_from_durable_receipt"
        except Exception as exc:
            failure_reason = f"started fit has untrusted durable bytes; no retry: {type(exc).__name__}: {exc}"
    else:
        failure_reason = "started fit has no durable validated gamma and receipt; no retry"
    offset._append_ledger({"event": "FIT_FAILED", "fit_id": fit_id,
                           "reservation_key": starts[0].get("reservation_key"),
                           "card_id": CARD_ID, "stage": STAGE,
                           "reason": failure_reason},
                          path=ledger_path, official=official)
    return "failed_no_retry"


def _expected_source_commit(freeze: Mapping[str, Any]) -> None:
    if _source_hashes() != freeze["source_hashes"]:
        raise PilotError("Frozen pilot source hashes changed after commit")
    commit = str(freeze["source_commit"])
    if _git("merge-base", "--is-ancestor", commit, "HEAD") is None:
        raise PilotError("Frozen pilot source commit is no longer an ancestor")
    if _sha_file(TRAIN) != freeze["input_sha256"]:
        raise PilotError("Official Train hash changed after freeze")


def freeze() -> dict[str, Any]:
    preflight_result = _json(REPORT_ROOT / "PREFLIGHT.json")
    synthetic = _json(REPORT_ROOT / "SYNTHETIC_QA.json")
    if preflight_result.get("status") != "PREFIT_PASS" or synthetic.get("status") != "PASS":
        raise PilotError("Target-free preflight and synthetic six-fit QA must pass before freeze")
    if _git("status", "--porcelain"):
        raise PilotError("Commit all source, recipe, tests, preflight and synthetic evidence before freeze")
    events, snapshot, ledger_sha = _current_ledger()
    if snapshot != PRE_FIT_SNAPSHOT or ledger_sha != preflight_result.get("live_ledger_sha256"):
        raise PilotError("Live fit ledger changed after target-free preflight")
    value = {
        "schema_version": 1, "card_id": CARD_ID, "stage": STAGE,
        "status": "FROZEN_BEFORE_FIT", "source_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"), "source_hashes": _source_hashes(),
        "recipe_sha256": _sha_file(REPORT_ROOT / "RECIPE.json"),
        "preflight_sha256": _sha_file(REPORT_ROOT / "PREFLIGHT.json"),
        "synthetic_qa_sha256": _sha_file(REPORT_ROOT / "SYNTHETIC_QA.json"),
        "input_sha256": preflight_result["input_sha256"],
        "live_ledger_sha256_before_reservation": ledger_sha,
        "live_ledger_snapshot_before_reservation": snapshot,
        "authorized_amendment_id": AMENDMENT_ID,
        "authorized_fit_ids": list(FIT_IDS),
        "origins": preflight_result["origins"],
        "target_columns_read": [], "query_labels_read": False,
        "test_opened": False, "sample_submission_opened": False, "kaggle_accessed": False,
    }
    value["freeze_signature"] = offset.canonical_sha256(value)
    _write_json(REPORT_ROOT / "FREEZE.json", value)
    _write_json(RUN_ROOT / "FREEZE.json", value)
    return value


def freeze_successor() -> dict[str, Any]:
    """Bind a prefit-only schema correction without changing or re-reserving slots."""
    old = _json(RUN_ROOT / "FREEZE.json")
    preflight_path = REPORT_ROOT / "PREFLIGHT_SUCCESSOR.json"
    synthetic_path = REPORT_ROOT / "SYNTHETIC_QA_SUCCESSOR.json"
    preflight_result = _json(preflight_path)
    synthetic = _json(synthetic_path)
    if (old.get("status") != "FROZEN_BEFORE_FIT"
            or preflight_result.get("status") != "PREFIT_PASS"
            or preflight_result.get("preflight_mode") != "PREFIT_SCHEMA_SUCCESSOR_NO_FITS_STARTED"
            or preflight_result.get("parent_freeze_signature") != old.get("freeze_signature")
            or preflight_result.get("cache_migration_verification", {}).get("status") != "PASS"
            or preflight_result.get("cache_migration_verification", {}).get("old_row_metadata_bytes_recovered") is not False
            or synthetic.get("status") != "PASS"):
        raise PilotError("Successor preflight/synthetic evidence does not match the preserved initial freeze")
    if _git("status", "--porcelain"):
        raise PilotError("Commit successor code and QA evidence before creating the successor freeze")
    events, snapshot, ledger_sha = _current_ledger()
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == RESERVATION_KEY]
    own_starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    _assert_successor_reservation_untouched(
        snapshot=snapshot, expected_snapshot=POST_RESERVE_SNAPSHOT,
        ledger_sha=ledger_sha, expected_ledger_sha=preflight_result.get("live_ledger_sha256"),
        reservations=reservations, expected_parent_signature=old["freeze_signature"],
        expected_parent_source=old["source_commit"], own_starts=own_starts)
    value = {
        "schema_version": 1, "card_id": CARD_ID, "stage": STAGE,
        "status": "SUCCESSOR_FREEZE_BEFORE_FIT",
        "successor_reason": "prefit ID arrays were reserialized from object to Unicode before any fit; original row-NPZ bytes unavailable, migrated values independently reconciled",
        "parent_freeze_signature": old["freeze_signature"],
        "parent_source_commit": old["source_commit"],
        "source_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "source_hashes": _source_hashes(),
        "recipe_sha256": _sha_file(REPORT_ROOT / "RECIPE.json"),
        "preflight_path": preflight_path.relative_to(ROOT).as_posix(),
        "preflight_sha256": _sha_file(preflight_path),
        "synthetic_qa_path": synthetic_path.relative_to(ROOT).as_posix(),
        "synthetic_qa_sha256": _sha_file(synthetic_path),
        "input_sha256": preflight_result["input_sha256"],
        "live_ledger_sha256_before_rebind": ledger_sha,
        "live_ledger_snapshot_before_rebind": snapshot,
        "authorized_amendment_id": AMENDMENT_ID,
        "authorized_fit_ids": list(FIT_IDS),
        "origins": preflight_result["origins"],
        "cache_migration_verification": preflight_result["cache_migration_verification"],
        "original_row_metadata_bytes_recovered": False,
        "original_row_metadata_sha256_by_origin": {
            key: old["origins"][key]["row_arrays_sha256"]["metadata_file"]
            for key in ("E67", "E75")
        },
        "target_columns_read": [], "query_labels_read": False,
        "test_opened": False, "sample_submission_opened": False, "kaggle_accessed": False,
    }
    value["freeze_signature"] = offset.canonical_sha256(value)
    _write_json(REPORT_ROOT / "FREEZE_SUCCESSOR.json", value)
    _write_json(ACTIVE_FREEZE, value)
    return value


def load_freeze() -> dict[str, Any]:
    freeze_path = ACTIVE_FREEZE if ACTIVE_FREEZE.is_file() else RUN_ROOT / "FREEZE.json"
    freeze_data = _json(freeze_path)
    body = dict(freeze_data)
    signature = body.pop("freeze_signature", None)
    if signature != offset.canonical_sha256(body):
        raise PilotError("Pilot freeze signature mismatch")
    _expected_source_commit(freeze_data)
    if freeze_data.get("authorized_fit_ids") != list(FIT_IDS):
        raise PilotError("Pilot freeze fit ID order does not match authorization")
    return freeze_data


def _assert_successor_reservation_untouched(*, snapshot: Mapping[str, Any],
                                            expected_snapshot: Mapping[str, Any],
                                            ledger_sha: str, expected_ledger_sha: str,
                                            reservations: list[Mapping[str, Any]],
                                            expected_parent_signature: str,
                                            expected_parent_source: str,
                                            own_starts: list[Mapping[str, Any]]) -> None:
    if (dict(snapshot) != dict(expected_snapshot) or ledger_sha != expected_ledger_sha
            or len(reservations) != 1
            or reservations[0].get("fit_ids") != list(FIT_IDS)
            or reservations[0].get("freeze_signature") != expected_parent_signature
            or reservations[0].get("source_commit") != expected_parent_source
            or own_starts):
        raise PilotError("Successor freeze requires the unchanged unique reservation and zero fit starts")


def bind_successor_freeze() -> dict[str, Any]:
    freeze_data = load_freeze()
    parent_signature = freeze_data.get("parent_freeze_signature")
    if not parent_signature:
        raise PilotError("Active freeze is not a successor freeze")
    events, snapshot, ledger_sha = _current_ledger()
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == RESERVATION_KEY]
    own_starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    _assert_successor_reservation_untouched(
        snapshot=snapshot, expected_snapshot=freeze_data["live_ledger_snapshot_before_rebind"],
        ledger_sha=ledger_sha, expected_ledger_sha=freeze_data["live_ledger_sha256_before_rebind"],
        reservations=reservations, expected_parent_signature=parent_signature,
        expected_parent_source=freeze_data["parent_source_commit"], own_starts=own_starts)
    existing = [e for e in events if e.get("event") == "PREFIT_FREEZE_SUCCESSOR_BOUND"
                and e.get("reservation_key") == RESERVATION_KEY]
    expected_event = {"event": "PREFIT_FREEZE_SUCCESSOR_BOUND", "reservation_key": RESERVATION_KEY,
                      "card_id": CARD_ID, "stage": STAGE, "fit_ids": list(FIT_IDS),
                      "parent_freeze_signature": parent_signature,
                      "freeze_signature": freeze_data["freeze_signature"],
                      "source_commit": freeze_data["source_commit"],
                      "reason": freeze_data["successor_reason"]}
    if existing:
        if len(existing) != 1 or any(existing[0].get(k) != v for k, v in expected_event.items()):
            raise PilotError("Conflicting successor binding event already exists")
        event = existing[0]
        idempotent = True
    else:
        event = offset._append_ledger(expected_event, path=LEDGER, official=True)
        idempotent = False
    after_events, after_snapshot, after_sha = _current_ledger()
    result = {"status": "BOUND", "idempotent_replay": idempotent,
              "event": event, "ledger_sha256": after_sha,
              "ledger_snapshot": after_snapshot,
              "no_fit_started": not any(e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS
                                         for e in after_events)}
    _write_json(REPORT_ROOT / "RESERVATION_REBIND.json", result)
    _write_json(RUN_ROOT / "RESERVATION_REBIND.json", result)
    return result


def reserve() -> dict[str, Any]:
    freeze_data = load_freeze()
    events, snapshot, ledger_sha = _current_ledger()
    if ledger_sha != freeze_data["live_ledger_sha256_before_reservation"] or snapshot != PRE_FIT_SNAPSHOT:
        raise PilotError("Live shared budget changed after prefit freeze")
    if any(e.get("reservation_key") == RESERVATION_KEY for e in events):
        raise PilotError("Reservation key already exists; refusing a second reservation")
    amendment = {"event": "BUDGET_CAP_AMENDMENT", "amendment_id": AMENDMENT_ID,
                 "card_id": CARD_ID, "stage": STAGE, "previous_global_cap": 143,
                 "global_cap": 149, "previous_caps": PRE_FIT_SNAPSHOT["effective_category_caps"],
                 "effective_caps": POST_RESERVE_CAPS, "added_fits": 6,
                 "source_commit": freeze_data["source_commit"],
                 "freeze_signature": freeze_data["freeze_signature"],
                 "authorization": "reports/optimization_v3/bigram_position_readiness_v1/SOL_PILOT_AUTHORIZATION.md"}
    offset._append_ledger(amendment, path=LEDGER, official=True)
    reservation = {"event": "CARD_FITS_RESERVED", "reservation_key": RESERVATION_KEY,
                   "card_id": CARD_ID, "stage": STAGE, "category": "pilot", "fits": 6,
                   "fit_ids": list(FIT_IDS), "amendment_id": AMENDMENT_ID,
                   "source_commit": freeze_data["source_commit"],
                   "freeze_signature": freeze_data["freeze_signature"]}
    offset._append_ledger(reservation, path=LEDGER, official=True)
    events, snapshot, ledger_sha = _current_ledger()
    expected = dict(PRE_FIT_SNAPSHOT)
    expected.update({"event_rows": PRE_FIT_SNAPSHOT["event_rows"] + 2, "global_cap": 149,
                     "effective_category_caps": POST_RESERVE_CAPS,
                     "active_reservations_by_category": {"continuation": 50, "control_replay": 2,
                                                          "pilot": 91, "technical_retry": 1},
                     "active_reservations": 144})
    if snapshot != expected:
        raise PilotError(f"Post-reservation shared accounting mismatch: {snapshot}")
    result = {"status": "RESERVED", "amendment": amendment, "reservation": reservation,
              "ledger_sha256": ledger_sha, "snapshot": snapshot}
    _write_json(RUN_ROOT / "RESERVATION.json", result)
    return result


def _load_cached_origin(origin: int):
    directory = RUN_ROOT / f"E{origin}"
    rows_path = directory / "frozen_rows.npz"
    with np.load(rows_path, allow_pickle=False) as archive:
        rows = {k: archive[k].copy() for k in archive.files}
    matrices = {}
    for arm in ARMS:
        name = arm.lower()
        matrices[name] = {
            "fit": sparse.load_npz(directory / f"fit_{name}.npz").tocsr(),
            "query": sparse.load_npz(directory / f"query_{name}.npz").tocsr(),
        }
    return rows, matrices


def _live_budget_for_fits(freeze_data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events, snapshot, _ = _current_ledger()
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservations) != 1 or reservations[0].get("fit_ids") != list(FIT_IDS):
        raise PilotError("Unique authorized six-fit reservation missing or malformed")
    parent_signature = freeze_data.get("parent_freeze_signature")
    if parent_signature:
        bound = [e for e in events if e.get("event") == "PREFIT_FREEZE_SUCCESSOR_BOUND"
                 and e.get("reservation_key") == RESERVATION_KEY]
        if (reservations[0].get("freeze_signature") != parent_signature
                or reservations[0].get("source_commit") != freeze_data.get("parent_source_commit")
                or len(bound) != 1 or bound[0].get("freeze_signature") != freeze_data["freeze_signature"]
                or bound[0].get("source_commit") != freeze_data["source_commit"]):
            raise PilotError("Successor freeze has not been bound to the original six-slot reservation")
    elif (reservations[0].get("freeze_signature") != freeze_data["freeze_signature"]
          or reservations[0].get("source_commit") != freeze_data["source_commit"]):
        raise PilotError("Reservation does not bind to current pilot freeze/source")
    if snapshot["global_cap"] != 149 or snapshot["effective_category_caps"] != POST_RESERVE_CAPS:
        raise PilotError("Live cap changed after pilot reservation")
    own_events = [e for e in events if e.get("fit_id") in FIT_IDS]
    if any(e.get("reservation_key") != RESERVATION_KEY for e in own_events):
        raise PilotError("Pilot fit ID appears under a different reservation")
    if snapshot["fit_starts"] > 144 or snapshot["fit_pending"] > 1:
        raise PilotError(f"Research Next cap/pending-fit invariant failed: {snapshot}")
    return events, snapshot


def _check_resources(*, initial: bool) -> dict[str, Any]:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    if rss > RSS_LIMIT:
        raise PilotError(f"Current process RSS exceeds 6 GiB: {rss}")
    minimum = INITIAL_FREE_MIN if initial else FREE_RESERVE
    if available < minimum:
        raise PilotError(f"Available memory below {'initial 4 GiB' if initial else '2 GiB reserve'}: {available}")
    busy = []
    for other in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if other.pid == os.getpid():
                continue
            name = (other.info.get("name") or "").lower()
            cmd = " ".join(other.info.get("cmdline") or []).lower()
            if "python" in name and any(token in cmd for token in ("optimization_v3_", "positional_", "bigram_position")):
                busy.append({"pid": other.pid, "name": name, "cmdline": cmd[:240]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if busy:
        raise PilotError(f"Another research fit process appears active; do not overlap: {busy}")
    return {"process_rss_bytes": rss, "available_memory_bytes": available,
            "initial_threshold_bytes": INITIAL_FREE_MIN, "reserve_threshold_bytes": FREE_RESERVE,
            "other_research_processes": busy}


def _fit_artifact_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "fits" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def _fit_spec(fit_id: str):
    if fit_id not in FIT_IDS:
        raise PilotError(f"Fit ID is not authorized: {fit_id}")
    parts = fit_id.split("_")
    origin = int(parts[-2][1:])
    arm = parts[-1]
    return origin, arm


def _fit_one(fit_id: str, freeze_data: Mapping[str, Any]) -> dict[str, Any]:
    origin, arm = _fit_spec(fit_id)
    events, _ = _live_budget_for_fits(freeze_data)
    existing_start = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    gamma_path, receipt_path = _fit_artifact_paths(fit_id)
    if terminal:
        if len(terminal) != 1:
            raise PilotError(f"Fit {fit_id} has duplicate terminal events")
        if terminal[0].get("event") == "FIT_FAILED":
            return {"fit_id": fit_id, "status": "FIT_FAILED", "reason": terminal[0].get("reason")}
        if not gamma_path.is_file() or not receipt_path.is_file():
            raise PilotError(f"Saved fit {fit_id} has no durable artifact/receipt")
        receipt = _json(receipt_path)
        if (receipt.get("freeze_signature") != freeze_data["freeze_signature"]
                or receipt.get("artifact_sha256") != _sha_file(gamma_path)):
            raise PilotError(f"Saved fit {fit_id} no longer matches its receipt")
        return {"fit_id": fit_id, "status": "FIT_SAVED", "artifact_sha256": receipt["artifact_sha256"]}
    if existing_start:
        _check_resources(initial=False)
        recovered = _recover_one(fit_id, ledger_path=LEDGER, official=True,
                                 fit_path=gamma_path, receipt_path=receipt_path,
                                 expected_freeze=freeze_data["freeze_signature"])
        return {"fit_id": fit_id, "status": recovered}
    if terminal or gamma_path.exists() or receipt_path.exists():
        raise PilotError(f"Fit {fit_id} has unexpected terminal/artifact state before start")
    already_started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    already_terminal_ids = {e.get("fit_id") for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    if len(already_started) != len(already_terminal_ids.intersection(FIT_IDS)):
        raise PilotError("A previous pilot fit is pending; resume it before starting another")
    resource = _check_resources(initial=(not already_started))
    rows, matrices = _load_cached_origin(origin)
    x = matrices[arm.lower()]["fit"]
    z = np.asarray(rows["z_fit_original"], dtype=np.float64)
    fit_rows = np.asarray(rows["fit_rows"], dtype=np.int64)
    if x.shape[0] != len(fit_rows) or len(z) != len(fit_rows):
        raise PilotError(f"E{origin} {arm} fit matrix/logit rows do not align")
    _check_resources(initial=False)
    event = {"event": "FIT_STARTED", "fit_id": fit_id,
             "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
             "origin": origin, "arm": arm, "source_commit": freeze_data["source_commit"],
             "freeze_signature": freeze_data["freeze_signature"], "resource_snapshot": resource}
    offset._append_ledger(event, path=LEDGER, official=True)
    try:
        # The first authorized read of artifact_signal occurs only after FIT_STARTED is durable.
        y = _read_fit_labels_only(TRAIN, fit_rows)
        with threadpool_limits(limits=2):
            gamma, optimizer = fit_offset_coefficients(x, z, y, c=1.0)
        if not optimizer.get("success") or not np.isfinite(gamma).all():
            raise PilotError("Frozen L-BFGS-B optimizer acceptance did not pass")
        _check_resources(initial=False)
        gamma_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        offset._atomic_npz(gamma_path, gamma=gamma)
        receipt = {
            "schema_version": 1, "fit_id": fit_id, "card_id": CARD_ID, "stage": STAGE,
            "origin": origin, "arm": arm, "source_commit": freeze_data["source_commit"],
            "freeze_signature": freeze_data["freeze_signature"], "input_sha256": freeze_data["input_sha256"],
            "artifact_sha256": _sha_file(gamma_path), "gamma_sha256": offset.sha256_array(gamma),
            "fit_rows": len(fit_rows), "fit_rows_sha256": offset.sha256_array(fit_rows),
            "fit_record_ids_sha256": offset.sha256_array(rows["fit_record_ids"].astype(str)),
            "fit_label_sha256": offset.sha256_array(y), "fit_label_count_0": int(np.count_nonzero(y == 0)),
            "fit_label_count_1": int(np.count_nonzero(y == 1)),
            "feature_matrix_sha256": offset.sha256_csr(x), "feature_width": x.shape[1],
            "offset_sha256": offset.sha256_array(z), "optimizer": optimizer,
            "target_boundary": "artifact_signal read after durable FIT_STARTED; fit prefix only",
            "query_labels_read": False, "test_opened": False,
        }
        offset._atomic_json(receipt_path, receipt)
        offset._append_ledger({"event": "FIT_SAVED", "fit_id": fit_id,
                               "reservation_key": RESERVATION_KEY, "card_id": CARD_ID,
                               "stage": STAGE, "source_commit": freeze_data["source_commit"],
                               "freeze_signature": freeze_data["freeze_signature"],
                               "artifact_sha256": receipt["artifact_sha256"],
                               "receipt_sha256": _sha_file(receipt_path)}, path=LEDGER, official=True)
        return {"fit_id": fit_id, "status": "FIT_SAVED", "optimizer": optimizer,
                "artifact_sha256": receipt["artifact_sha256"], "receipt_sha256": _sha_file(receipt_path)}
    except Exception as exc:
        current = offset._ledger_events(LEDGER)
        term = [e for e in current if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
        if not term:
            offset._append_ledger({"event": "FIT_FAILED", "fit_id": fit_id,
                                   "reservation_key": RESERVATION_KEY, "card_id": CARD_ID,
                                   "stage": STAGE, "source_commit": freeze_data["source_commit"],
                                   "freeze_signature": freeze_data["freeze_signature"],
                                   "reason": f"single authorized attempt failed: {type(exc).__name__}: {exc}"},
                                  path=LEDGER, official=True)
        raise


def run_all_fits() -> list[dict[str, Any]]:
    freeze_data = load_freeze()
    _, snapshot = _live_budget_for_fits(freeze_data)
    if not (RUN_ROOT / "RESERVATION.json").is_file():
        raise PilotError("Official six-fit reservation artifact is missing")
    results = []
    for index, fit_id in enumerate(FIT_IDS):
        # exactly one gamma estimator process at a time, in frozen order
        if index == 0:
            _check_resources(initial=True)
        try:
            results.append(_fit_one(fit_id, freeze_data))
        except Exception as exc:
            live = offset._ledger_events(LEDGER)
            terminal = [e for e in live if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                        and e.get("fit_id") == fit_id]
            if len(terminal) == 1 and terminal[0].get("event") == "FIT_FAILED":
                results.append({"fit_id": fit_id, "status": "FIT_FAILED",
                                "reason": terminal[0].get("reason"),
                                "exception": f"{type(exc).__name__}: {exc}"})
                continue
            raise
    events, snapshot, ledger_sha = _current_ledger()
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("fit_id") in FIT_IDS]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("fit_id") in FIT_IDS]
    started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") in FIT_IDS]
    if len(started) != 6 or len(saved) + len(failed) != 6 or snapshot["fit_pending"]:
        raise PilotError(f"Six-fit ledger did not reconcile cleanly: {snapshot}")
    result = {"status": "ALL_SAVED" if len(saved) == 6 else "FIT_FAILURES",
              "fit_ids": list(FIT_IDS), "starts": len(started), "saved": len(saved),
              "failed": len(failed), "snapshot": snapshot, "ledger_sha256": ledger_sha,
              "fits": results, "query_labels_read": False}
    _write_json(RUN_ROOT / "ACCOUNTING.json", result)
    if len(saved) != 6:
        (REPORT_ROOT / "RESULTS.md").write_text(
            "# CLS_BIGRAM_POSITION_PILOT_V1 — fit failure checkpoint\n\n"
            "At least one single-attempt fit did not save a validated gamma. No retry was made. "
            "All six authorized fit IDs were reconciled in the shared ledger. Query labels were not opened, "
            "and no temporal metrics were evaluated. See `experiments/runs/optimization_v3_research_next_v1/"
            "BIGRAM_POSITION_PILOT_V1/ACCOUNTING.json` and the per-fit receipts.\n",
            encoding="utf-8")
    return result


def build_predictions() -> dict[str, Any]:
    freeze_data = load_freeze()
    events, snapshot = _live_budget_for_fits(freeze_data)
    saved_ids = {e.get("fit_id") for e in events if e.get("event") == "FIT_SAVED"}
    if not set(FIT_IDS).issubset(saved_ids):
        raise PilotError("All six authorized gamma receipts must be durably saved before prediction")
    predictions = {}
    for origin in ORIGINS:
        rows, matrices = _load_cached_origin(origin)
        present = np.asarray(rows["present_mask"], dtype=bool)
        no_known = np.asarray(rows["no_known_bigram"], dtype=bool)
        z = np.asarray(rows["z_query_original"], dtype=np.float64)
        p_a02 = np.asarray(rows["p_a02"], dtype=np.float64)
        p_p = np.asarray(rows["p_p"], dtype=np.float64)
        mix = np.asarray(rows["mix50"], dtype=np.float64)
        arm_arrays = {}
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            gamma_path, receipt_path = _fit_artifact_paths(fit_id)
            receipt = _json(receipt_path)
            if receipt.get("freeze_signature") != freeze_data["freeze_signature"] or receipt.get("artifact_sha256") != _sha_file(gamma_path):
                raise PilotError(f"E{origin} {arm} receipt is not bound to the frozen artifact")
            with np.load(gamma_path, allow_pickle=False) as a:
                gamma = np.asarray(a["gamma"], dtype=np.float64).copy()
            xq = matrices[arm.lower()]["query"]
            q, route = route_candidate(xq, z, gamma, present_mask=present, no_known_bigram=no_known,
                                       p_pos_original=expit(z), p_a02=p_a02, p_p=p_p,
                                       mix50_original=mix)
            arm_arrays[f"q_{arm.lower()}"] = q
            arm_arrays[f"route_{arm.lower()}"] = route
        path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        offset._atomic_npz(path, origin=np.asarray([origin], dtype=np.int16),
                           score_rows=rows["score_rows"], score_record_ids=rows["score_record_ids"],
                           score_cycles=rows["score_cycles"], present_mask=present,
                           no_known_bigram=no_known, p_p=p_p, p_a02=p_a02, mix50=mix,
                           **arm_arrays)
        predictions[f"E{origin}"] = {"path": path.relative_to(ROOT).as_posix(),
                                      "sha256": _sha_file(path), "rows": len(rows["score_rows"]),
                                      "record_ids_sha256": offset.sha256_array(rows["score_record_ids"].astype(str)),
                                      "score_cycles_sha256": offset.sha256_array(rows["score_cycles"].astype(np.int64))}
    ready = {"schema_version": 1, "card_id": CARD_ID,
             "freeze_signature": freeze_data["freeze_signature"],
             "source_commit": freeze_data["source_commit"], "input_sha256": freeze_data["input_sha256"],
             "query_labels_opened": False, "test_opened": False,
             "predictions": predictions}
    _write_json(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    return ready


def verify_prelabel() -> dict[str, Any]:
    freeze_data = load_freeze()
    from scripts.verify_optimization_v3_bigram_position_pilot_v1 import verify_prelabel_outputs
    result = verify_prelabel_outputs(ROOT, RUN_ROOT, freeze_data)
    if result.get("status") != "PASS" or result.get("query_labels_read") is not False:
        raise PilotError(f"Independent prelabel verifier did not PASS: {result}")
    result["freeze_signature"] = freeze_data["freeze_signature"]
    _write_json(RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
    _write_json(REPORT_ROOT / "PRELABEL_VERIFICATION.json", result)
    return result


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int8)
    return {"n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
            "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
            "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def _gate(name: str, value: float, *, minimum: float | None = None,
          maximum: float | None = None) -> dict[str, Any]:
    tolerance = 1e-12
    if (minimum is None) == (maximum is None):
        raise ValueError("a gate requires exactly one of minimum or maximum")
    if minimum is not None:
        passed = float(value) + tolerance >= float(minimum)
        bound: dict[str, float] = {"minimum": float(minimum)}
    else:
        passed = float(value) <= float(maximum) + tolerance
        bound = {"maximum": float(maximum)}
    return {"gate": name, "value": float(value), **bound, "tolerance": tolerance,
            "pass": bool(passed)}


def _views(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {"W4": (cycles >= 68) & (cycles <= 71),
                "W5": (cycles >= 72) & (cycles <= 75),
                "W6": (cycles >= 76) & (cycles <= 79),
                "H12_E67": (cycles >= 68) & (cycles <= 79)}
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def _groups(rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    present = np.asarray(rows["present_mask"], dtype=bool)
    ef = np.asarray(rows["entry_friction"]).astype(str)
    tl = np.asarray(rows["text_load_group"], dtype=np.int8)
    return {"availability_present": present, "availability_missing": ~present,
            "entry_friction_0": ef == "0", "entry_friction_1": ef == "1",
            **{f"text_load_Q{i+1}": tl == i for i in range(4)}}


def _write_plots(result: Mapping[str, Any]) -> list[str]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = ["MIX50", "BAG", "REAL", "SHUFFLED"]
    selected = result["origins"]["E67"]["views"]["H12_E67"]["metrics"]
    vals = [selected[k]["macro_f1"] for k in ("mix50", "bag", "real", "shuffled")]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    colors = ["#52616b", "#688fbd", "#2b8a6e", "#b07aa1"]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(max(0, min(vals) - 0.035), min(1, max(vals) + 0.025))
    ax.set_ylabel("Macro F1")
    ax.set_title("E67 H12: frozen baseline and three bigram arms")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p1 = REPORT_ROOT / "primary_comparison.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    paths.append(p1.relative_to(ROOT).as_posix())
    rows = []
    for origin in ("E67", "E75"):
        for view, block in result["origins"][origin]["views"].items():
            rows.append((f"{origin} {view}", block["deltas"]["real_minus_mix50"]))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    names = [x[0] for x in rows]
    deltas = [x[1] for x in rows]
    ax.bar(names, deltas, color=["#2b8a6e" if v >= 0 else "#c44e52" for v in deltas])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("REAL − MIX50 Macro F1")
    ax.set_title("Temporal view sensitivity; E75 W6 overlaps E67 W6")
    ax.tick_params(axis="x", rotation=22)
    fig.tight_layout()
    p2 = REPORT_ROOT / "temporal_deltas.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    paths.append(p2.relative_to(ROOT).as_posix())
    return paths


def score() -> dict[str, Any]:
    freeze_data = load_freeze()
    verifier = _json(RUN_ROOT / "PRELABEL_VERIFICATION.json")
    if (verifier.get("status") != "PASS" or verifier.get("query_labels_read") is not False
            or verifier.get("freeze_signature") != freeze_data["freeze_signature"]):
        raise PilotError("Current independent prelabel PASS is required before opening query labels")
    results = {"schema_version": 1, "card_id": CARD_ID,
               "freeze_signature": freeze_data["freeze_signature"],
               "source_commit": freeze_data["source_commit"], "input_sha256": freeze_data["input_sha256"],
               "query_labels_opened_after_independent_prelabel_pass": True,
               "origins": {}, "gates": [], "status": "DIAGNOSTIC_ONLY"}
    # First label access for the query windows; no Test or SampleSubmission access.
    labels = pd.read_csv(TRAIN, usecols=["artifact_signal"])["artifact_signal"]
    y_all = pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.int8)
    del labels
    for origin in ORIGINS:
        rows, matrices = _load_cached_origin(origin)
        pred_path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        with np.load(pred_path, allow_pickle=False) as a:
            pred = {k: a[k].copy() for k in a.files}
        score_rows = np.asarray(rows["score_rows"], dtype=np.int64)
        y = y_all[score_rows]
        cycles = np.asarray(rows["score_cycles"], dtype=np.int64)
        if not np.array_equal(pred["score_record_ids"].astype(str), rows["score_record_ids"].astype(str)):
            raise PilotError(f"E{origin} prediction IDs changed immediately before metric computation")
        arm_prob = {arm.lower(): np.asarray(pred[f"route_{arm.lower()}"], dtype=np.float64) for arm in ARMS}
        arm_prob["mix50"] = np.asarray(pred["mix50"], dtype=np.float64)
        view_metrics = {}
        per_cycle = []
        subgroup_metrics = []
        for view, mask in _views(origin, cycles).items():
            metrics = {arm: _metric(y[mask], p[mask]) for arm, p in arm_prob.items()}
            deltas = {f"{arm}_minus_mix50": metrics[arm]["macro_f1"] - metrics["mix50"]["macro_f1"]
                      for arm in ("bag", "real", "shuffled")}
            deltas["real_minus_bag"] = metrics["real"]["macro_f1"] - metrics["bag"]["macro_f1"]
            deltas["real_minus_shuffled"] = metrics["real"]["macro_f1"] - metrics["shuffled"]["macro_f1"]
            view_metrics[view] = {"rows": int(mask.sum()), "metrics": metrics, "deltas": deltas}
            if (origin == 67 and view in {"H12_E67"}) or (origin == 75 and view == "W6"):
                for comparator in ("mix50", "bag", "shuffled"):
                    delta = metrics["real"]["macro_f1"] - metrics[comparator]["macro_f1"]
                    if comparator == "mix50":
                        limit = 0.0005
                    else:
                        limit = 0.0
                    results["gates"].append(_gate(
                        f"E{origin}_{view}_REAL_gain_vs_{comparator}", delta, minimum=limit))
            if origin == 67 and view in {"W4", "W5", "W6"}:
                delta = metrics["real"]["macro_f1"] - metrics["mix50"]["macro_f1"]
                results["gates"].append(_gate(f"E67_{view}_REAL_harm_vs_MIX50_max_0.003",
                                               delta, minimum=-0.003))
            for group, group_mask in _groups(rows).items():
                use = mask & group_mask
                n = int(use.sum())
                positives = int(y[use].sum())
                negatives = n - positives
                supported = n >= 200 and positives >= 30 and negatives >= 30
                group_result = {"origin": origin, "view": view, "group": group,
                                "rows": n, "positive": positives, "negative": negatives,
                                "supported": supported, "status": "EVALUATED" if supported else "UNKNOWN",
                                "metrics": {arm: _metric(y[use], p[use]) for arm, p in arm_prob.items()}}
                if supported:
                    base = group_result["metrics"]["mix50"]
                    real = group_result["metrics"]["real"]
                    f1_harm = base["macro_f1"] - real["macro_f1"]
                    recall_harm = base["positive_recall"] - real["positive_recall"]
                    group_result["real_f1_harm_vs_mix50"] = f1_harm
                    group_result["real_positive_recall_harm_vs_mix50"] = recall_harm
                    f1_gate = _gate(f"E{origin}_{view}_{group}_F1_harm_max_0.01", f1_harm, maximum=0.01)
                    recall_gate = _gate(f"E{origin}_{view}_{group}_positive_recall_harm_max_0.02",
                                        recall_harm, maximum=0.02)
                    group_result["f1_gate_pass"] = f1_gate["pass"]
                    group_result["positive_recall_gate_pass"] = recall_gate["pass"]
                    results["gates"].extend([
                        f1_gate, recall_gate,
                    ])
                subgroup_metrics.append(group_result)
        for cycle in sorted(np.unique(cycles).tolist()):
            cm = cycles == cycle
            per_cycle.append({"cycle_code": int(cycle), "rows": int(cm.sum()),
                              "metrics": {arm: _metric(y[cm], p[cm]) for arm, p in arm_prob.items()}})
        results["origins"][f"E{origin}"] = {"fit_rows": int(len(rows["fit_rows"])),
                                               "query_rows": int(len(score_rows)),
                                               "query_cycles": [int(cycles.min()), int(cycles.max())],
                                               "views": view_metrics, "subgroups": subgroup_metrics,
                                               "per_cycle": per_cycle}
    results["gates_pass"] = bool(all(g.get("pass", False) for g in results["gates"]))
    results["unsupported_subgroups"] = [
        {"origin": item["origin"], "view": item["view"], "group": item["group"], "rows": item["rows"],
         "positive": item["positive"], "negative": item["negative"]}
        for block in results["origins"].values() for item in block["subgroups"] if not item["supported"]]
    results["gate_status"] = ("PASS" if results["gates_pass"] and not results["unsupported_subgroups"]
                               else "FAIL" if not results["gates_pass"] else "UNKNOWN_UNSUPPORTED_SUBGROUPS")
    results["interpretation"] = (
        "Diagnostic temporal development evidence only. E67 and E75 query windows overlap; "
        "gate outcomes do not establish statistical significance, causality, or untouched-holdout generalization."
    )
    results["access_boundary"] = {"classification_train_target_opened": True,
                                  "test_opened": False, "sample_submission_opened": False,
                                  "kaggle_accessed": False}
    _write_json(RUN_ROOT / "RESULTS.json", results)
    plots = _write_plots(results)
    _write_report(results, plots)
    return results


def _write_report(result: Mapping[str, Any], plots: list[str]) -> None:
    lines = ["# CLS_BIGRAM_POSITION_PILOT_V1 — hasil pilot", "",
             f"**Status gate:** {result['gate_status']} (engineering criteria; diagnostic-only)", "",
             "Pertanyaan: apakah posisi bigram yang dikenal memberi tambahan prediksi dibandingkan bag bigram dan shuffled-position, di atas logit C2B + POS unigram yang dibekukan?", "",
             "## Hasil per temporal view", "",
             "Macro F1 ditampilkan untuk MIX50 incumbent, BAG, REAL, dan SHUFFLED. Semua precision/recall/confusion/ROC-AUC dan denominator tersedia di `RESULTS.json`.", "",
             "| Origin | View | n | MIX50 | BAG | REAL | SHUFFLED | REAL − MIX50 | REAL − BAG | REAL − SHUFFLED |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for origin, block in result["origins"].items():
        for view, item in block["views"].items():
            m, d = item["metrics"], item["deltas"]
            lines.append(f"| {origin} | {view} | {item['rows']} | {m['mix50']['macro_f1']:.6f} | {m['bag']['macro_f1']:.6f} | {m['real']['macro_f1']:.6f} | {m['shuffled']['macro_f1']:.6f} | {d['real_minus_mix50']:+.6f} | {d['real_minus_bag']:+.6f} | {d['real_minus_shuffled']:+.6f} |")
    lines.extend(["", "## Keputusan gate", "", "| Gate | Nilai | Batas | Hasil |", "|---|---:|---:|---|"])
    for gate in result["gates"]:
        limit = gate.get("minimum", gate.get("maximum"))
        lines.append(f"| {gate['gate']} | {gate['value']:+.6f} | {limit:+.6f} | {'PASS' if gate['pass'] else 'FAIL'} |")
    lines.extend(["", "## Interpretasi dan batas", "",
                  "REAL dibandingkan dengan BAG dan SHUFFLED untuk membedakan nilai posisi dari frekuensi pasangan dan dari urutan acak yang mempertahankan bag serta ukuran bin. Hasil E67/E75 adalah development evidence dengan jendela yang tumpang tindih. Gate bukan uji signifikansi.", "",
                  "Semua subgroup yang tidak memenuhi dukungan n≥200 dan masing-masing kelas≥30 dilaporkan UNKNOWN. Tidak ada klaim kausal atau semantik. Tidak ada Test, SampleSubmission, atau Kaggle access.", "",
                  "## Visual", ""])
    for path in plots:
        lines.append(f"![{Path(path).stem}]({path})")
        lines.append("")
    (REPORT_ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "synthetic", "freeze", "reserve",
                                             "freeze-successor", "bind-successor",
                                             "run-all", "predictions", "verify-prelabel", "score"))
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight": result = preflight()
        elif args.command == "synthetic":
            if args.temp_root is None:
                raise PilotError("synthetic requires --temp-root under system temporary storage")
            result = run_synthetic_lifecycle(args.temp_root)
            target = (REPORT_ROOT / "SYNTHETIC_QA_SUCCESSOR.json"
                      if (REPORT_ROOT / "PREFLIGHT_SUCCESSOR.json").is_file()
                      else REPORT_ROOT / "SYNTHETIC_QA.json")
            _write_json(target, result)
        elif args.command == "freeze": result = freeze()
        elif args.command == "reserve": result = reserve()
        elif args.command == "freeze-successor": result = freeze_successor()
        elif args.command == "bind-successor": result = bind_successor_freeze()
        elif args.command == "run-all": result = run_all_fits()
        elif args.command == "predictions": result = build_predictions()
        elif args.command == "verify-prelabel": result = verify_prelabel()
        else: result = score()
        print(json.dumps(offset._jsonable(result), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
