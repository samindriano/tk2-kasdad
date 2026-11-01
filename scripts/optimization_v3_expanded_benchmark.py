"""Seven-candidate V3 expansion: frozen continuation and audit runner.

This runner extends the existing C16/R22 checkpoint with C17, C11, R21, R11,
and original R06 only. It never changes pilot sources, never reads Test, and
keeps estimator fits serialized under the shared V3 training lock.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_batch1 import (  # noqa: E402
    append_event,
    exclusive_training_slot,
    file_sha256,
    json_sha256,
    read_bundle,
    reserve_card_budget,
    write_bundle_exclusive,
    write_json_exclusive,
)
from src.models.optimization_v3_expanded_continuation import (  # noqa: E402
    FIT_FUNCTIONS,
    NEW_CUTOFFS,
)


CAMPAIGN_ID = "optimization_v3_expanded_benchmark_v1"
ALL_CANDIDATES = ("C16", "C17", "C11", "R22_ORIGINAL", "R21", "R11", "R06_ORIGINAL")
NEW_CANDIDATES = ("C17", "C11", "R21", "R11", "R06")
FIT_ORDER = ("C17", "C11", "R21", "R11", "R06")
PILOT_CUTOFFS = (55, 67)
ALL_ORIGINS = (55, 59, 63, 67, 71, 75)
WINDOWS = {
    "W1": (55, 56, 59), "W2": (59, 60, 63), "W3": (63, 64, 67),
    "W4": (67, 68, 71), "W5": (71, 72, 75), "W6": (75, 76, 79),
}
EXPECTED_PRIMARY_ROWS = 38_588
EXPECTED_H12_ROWS = 38_588
EXPECTED_LATE_ROWS = 15_953
GLOBAL_CAP = 180
NEW_CONTINUATION_FITS = 52
NEW_FULL_TRAIN_FITS = 13
PREVIOUS_NEW_FULL_TRAIN_FITS = 2
RETRY_HEADROOM = 11
NEW_RESERVATION_KEY = "EXPANDED_BENCHMARK:C17_C11_R21_R11_R06"
NEW_RESERVATION_CARD = "C17+C11+R21+R11+R06"
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
TRAINING_LOCK = ROOT / "experiments/runs/optimization_v3_batch1/.training.lock"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/expanded_benchmark_v1"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_expanded_benchmark_v1"
REPORT_DIR = ROOT / "reports/optimization_v3/expanded_benchmark_v1"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
CAMPAIGN_REPORT_PATH = REPORT_DIR / "CAMPAIGN_REPORT.json"

ACTIVE_CARD_FILES = {
    "C17": ("experiments/optimization_v3/batch1/C17/recipe.json",
            "experiments/optimization_v3/batch1/C17/freeze.json"),
    "C11": ("experiments/optimization_v3/batch1/C11/recipe.json",
            "experiments/optimization_v3/batch1/C11/freeze.json"),
    "R21": ("experiments/optimization_v3/batch1/R21/recipe.json",
            "experiments/optimization_v3/batch1/R21/freeze.json"),
    "R11": ("experiments/optimization_v3/batch1/R11/revision_03/recipe.json",
            "experiments/optimization_v3/batch1/R11/revision_03/freeze.json"),
    "R06": ("experiments/optimization_v3/batch1/R06/recipe_revision_02.json",
            "experiments/optimization_v3/batch1/R06/freeze_revision_02.json"),
}


class ExpandedCampaignError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite value cannot be written to the frozen campaign")
    return value


def _expected_rows(cycles: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.flatnonzero((cycles >= int(low)) & (cycles <= int(high))).astype(np.int64)


def _source_path(raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def _active_card_artifacts() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for card, (recipe_rel, freeze_rel) in ACTIVE_CARD_FILES.items():
        recipe_path, freeze_path = ROOT / recipe_rel, ROOT / freeze_rel
        recipe, freeze = _read_json(recipe_path), _read_json(freeze_path)
        if freeze.get("signature") != json_sha256(freeze.get("identity", {})):
            raise ExpandedCampaignError(f"{card} active pilot freeze signature is invalid")
        output[card] = {
            "recipe": recipe, "freeze": freeze,
            "recipe_path": recipe_path, "freeze_path": freeze_path,
            "recipe_sha256": file_sha256(recipe_path), "freeze_sha256": file_sha256(freeze_path),
        }
    if output["R06"]["freeze"]["signature"] != "02ae1b4e6e6424a8ac7f66fd0f1b955e478f6265c4f071989d9d8c5ef532c0a8":
        raise ExpandedCampaignError("R06 original pilot lineage must use active revision 02")
    r06_policy = str(output["R06"]["recipe"].get("recipe", {}).get("policy", ""))
    if "0.75" not in r06_policy or "rank50(raw R06)" not in r06_policy or "present" in r06_policy.lower():
        raise ExpandedCampaignError("R06 recipe drifted from the original all-row rank50 blend")
    return output


def _audit_shared_engine_rebinding(expected_old_hash: str) -> dict[str, Any]:
    path = "src/models/optimization_v3_batch1.py"
    old = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"5840f4c^:{path}"],
        stderr=subprocess.DEVNULL,
    )
    new = (ROOT / path).read_bytes()
    old_hash, new_hash = hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest()
    if old_hash != expected_old_hash or new_hash != file_sha256(ROOT / path):
        raise ExpandedCampaignError("Shared batch-engine historical source hash mismatch")
    old_tree, new_tree = ast.parse(old.decode("utf-8")), ast.parse(new.decode("utf-8"))
    old_defs = {node.name: ast.dump(node, include_attributes=False)
                for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    new_defs = {node.name: ast.dump(node, include_attributes=False)
                for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    changed = sorted(name for name in set(old_defs) | set(new_defs) if old_defs.get(name) != new_defs.get(name))
    if changed != ["reserve_card_budget"]:
        raise ExpandedCampaignError(f"C17/R21 shared-engine source drift is not metadata-only: {changed}")
    return {"path": path, "historical_sha256": old_hash, "current_sha256": new_hash,
            "changed_top_level_definitions": changed,
            "interpretation": "reservation identity metadata only; no model, preprocessing, metric, or prediction code changed"}


def _load_contexts() -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts import optimization_v3_final_continuation_cls as cls_context
    from scripts import optimization_v3_final_continuation_r22 as reg_context
    classification = cls_context._context()
    regression = reg_context.prepare_context()
    if (len(classification["frame"]) != 50_608 or len(regression["frame"]) != 50_608
            or not np.array_equal(classification["cycles"], regression["cycles"])):
        raise ExpandedCampaignError("Official Train row/cycle population differs from frozen 50,608-row protocol")
    return classification, regression


def _active_pilot_dir(card: str) -> Path:
    path = ROOT / "experiments/runs/optimization_v3_batch1" / card
    return path / "revision_03" if card == "R11" else path


def _check_pilot_bundle(card: str, cutoff: int, frame: pd.DataFrame,
                        cycles: np.ndarray, controls: Mapping[str, Any],
                        card_artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    active = card_artifacts[card]
    path = _active_pilot_dir(card) / f"{card}_fit{cutoff}.npz"
    expected_rows = _expected_rows(cycles, cutoff + 1, cutoff + 12)
    expected_identity = {"card_id": card, "cutoff": int(cutoff),
                         "freeze_signature": active["freeze"]["signature"],
                         "score_rows": expected_rows.tolist()}
    arrays, sidecar = read_bundle(path, expected_identity)
    rows = np.asarray(arrays["score_rows"], dtype=np.int64)
    ids = np.asarray(arrays["score_record_ids"], dtype=str)
    if (not np.array_equal(rows, expected_rows)
            or not np.array_equal(ids, frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str))
            or len(np.unique(ids)) != len(ids)):
        raise ExpandedCampaignError(f"{card} pilot {cutoff} row/ID alignment failed")
    prediction = np.asarray(arrays["prediction"], dtype=np.float64)
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    if prediction.shape != rows.shape or raw.shape != rows.shape or baseline.shape != rows.shape:
        raise ExpandedCampaignError(f"{card} pilot {cutoff} predictions have invalid shapes")
    if not all(np.isfinite(x).all() for x in (prediction, raw, baseline)):
        raise ExpandedCampaignError(f"{card} pilot {cutoff} contains nonfinite outputs")

    if card in {"C17", "C11"}:
        control = {k: np.asarray(v) for k, v in controls.items()}
        if not np.array_equal(np.asarray(control["control_rows"], dtype=np.int64), rows):
            raise ExpandedCampaignError(f"{card} pilot {cutoff} control row identity differs")
        if not np.allclose(baseline, control["baseline_prediction"], atol=1e-12, rtol=0):
            raise ExpandedCampaignError(f"{card} pilot {cutoff} comparator differs from frozen baseline")
        missing = np.asarray(control["stream_missing"], dtype=bool)
        rebuilt = np.where(missing, control["cb_probability"],
                           0.75 * control["c2b_probability"] + 0.25 * raw)
        if not np.array_equal(prediction, rebuilt):
            raise ExpandedCampaignError(f"{card} pilot {cutoff} original route does not reconstruct")
        record_controls = control
    else:
        record_controls = {k: np.asarray(v) for k, v in controls.items()}
        if not np.allclose(baseline, record_controls["baseline_prediction"], atol=1e-10, rtol=0):
            raise ExpandedCampaignError(f"{card} pilot {cutoff} champion comparator differs")
        mapped = _rank50_independent(raw, cycles[rows])
        rebuilt = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
        if not np.allclose(prediction, rebuilt, atol=1e-10, rtol=0):
            raise ExpandedCampaignError(f"{card} pilot {cutoff} original rank50 blend does not reconstruct")
        record_controls["control_rows"] = rows
        record_controls["cycle_code"] = cycles[rows]
    metadata = sidecar.get("metadata", {})
    saved_state = metadata.get("state", {})
    if (metadata.get("card_id") != card or int(metadata.get("cutoff", -1)) != cutoff
            or bool(saved_state.get("test_data_used", False))
            or bool(saved_state.get("validation_labels_passed_to_fit", False))):
        raise ExpandedCampaignError(f"{card} pilot {cutoff} metadata lineage is invalid")
    return {
        "score_rows": rows, "score_record_ids": ids, "prediction": prediction,
        "raw_candidate": raw, "baseline_prediction": baseline,
        "control_arrays": record_controls, "metadata": metadata,
        "bundle_sha256": file_sha256(path), "sidecar_sha256": file_sha256(path.with_suffix(".json")),
        "bundle_identity": expected_identity,
    }


def _rank50_independent(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.ndim != 1 or raw.shape != cycles.shape or not len(raw) or not np.isfinite(raw).all():
        raise ExpandedCampaignError("Rank50 requires aligned finite predictions and cycle codes")
    mapped = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        values = raw[mask]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
            start = stop
        pct = 100.0 * ranks / len(values)
        mapped[mask] = 0.5 * values + 0.5 * pct
    return np.clip(mapped, 0.0, 100.0)


def _ledger_budget(*, expected_reserved: int | None = None) -> dict[str, Any]:
    reservations: dict[str, tuple[int, str]] = {}
    events = []
    if not LEDGER_PATH.is_file():
        raise ExpandedCampaignError("Active fit ledger is missing")
    for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExpandedCampaignError(f"Invalid runtime ledger line {line_number}") from exc
        events.append(event)
        if event.get("event") == "BUDGET_RESERVED":
            key = str(event["budget_key"])
            value = (int(event["predictor_fits"]), str(event.get("card_id", "")))
            if key in reservations:
                raise ExpandedCampaignError(f"Duplicate budget reservation {key}")
            reservations[key] = value
    reserved = sum(value[0] for value in reservations.values())
    if expected_reserved is not None and reserved != int(expected_reserved):
        raise ExpandedCampaignError(
            f"Expected active {int(expected_reserved)}-fit reservation checkpoint, observed {reserved}"
        )
    failures = [e for e in events if e.get("event") in {
        "FIT_FAILED", "FIT_INTERRUPTED_RESOURCE_PRESSURE", "PILOT_DISPATCH_FAILED_BEFORE_FIT",
        "CONTINUATION_FIT_FAILED", "EXPANDED_CONTINUATION_FIT_FAILED",
    }]
    return {
        "path": LEDGER_PATH.relative_to(ROOT).as_posix(), "sha256": file_sha256(LEDGER_PATH),
        "event_rows": len(events), "reservation_count": len(reservations),
        "reserved_predictor_fits": reserved, "remaining_reserved_capacity": GLOBAL_CAP - reserved,
        "reservation_keys": {k: list(v) for k, v in reservations.items()},
        "observed_failure_events": len(failures),
        "failed_attempts_and_interruptions": failures,
    }


def _submission_reuse_audit() -> dict[str, Any]:
    path = ROOT / "reports/submissions/c16_r22/SUBMISSION_STATUS.json"
    status = _read_json(path)
    if status.get("status") != "PREPARED_BLOCKED_AUTH_AND_LIVE_PREFLIGHT":
        raise ExpandedCampaignError("Existing C16/R22 submission status is not the expected prepared state")
    if status.get("live_kaggle_preflight", {}).get("upload_attempted") is not False:
        raise ExpandedCampaignError("C16/R22 prior upload status is ambiguous")
    output: dict[str, Any] = {"status_path": path.relative_to(ROOT).as_posix(),
                              "status_sha256": file_sha256(path), "status": status["status"],
                              "submission_refs": {"C16": None, "R22_ORIGINAL": None},
                              "upload_attempted": False, "candidates": {}}
    for task in ("classification", "regression"):
        record = status[task]
        csv_path = ROOT / record["csv"]["path_gitignored"]
        if not csv_path.is_file() or file_sha256(csv_path) != record["csv"]["sha256"]:
            raise ExpandedCampaignError(f"Prepared {record['candidate']} CSV is missing or hash-mismatched")
        state = record["fit_state"]
        validation_flags = ("validation_labels_passed_to_fit", "validation_data_passed_to_fit")
        test_flags = (name for name in ("test_data_used", "test_data_used_for_fit") if name in state)
        present_test_flags = tuple(test_flags)
        if (int(state.get("fit_rows", 0)) != 50_608
                or any(state.get(flag) is not False for flag in validation_flags)
                or not present_test_flags
                or any(state.get(flag) is not False for flag in present_test_flags)):
            raise ExpandedCampaignError(f"Prepared {record['candidate']} full-Train fit audit is invalid")
        output["candidates"][record["candidate"]] = {
            "csv_sha256": record["csv"]["sha256"], "rows": int(record["csv"]["rows"]),
            "predictor_fits": 1, "fit_rows": int(state["fit_rows"]),
            "reused_full_train_components": _json_clean(record.get("reused_full_train_components",
                                                               state.get("reused_full_train_components", {}))),
            "status": record["status"],
        }
    return output


def _r06_resource_estimate(frame: pd.DataFrame, cycles: np.ndarray, cutoff: int,
                           available: int) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scripts.optimization_v3_batch1_r06 import (
        CHAR_TFIDF_PARAMETERS, FREE_MEMORY_RESERVE_BYTES, JOB_MEMORY_CAP_BYTES,
        LIGHTGBM_MEMORY_OVERHEAD_BYTES, STRUCTURED_NNZ_UPPER_PER_ROW, THREADS as R06_THREADS,
    )
    fit_rows = np.flatnonzero(cycles <= cutoff)
    fit_x = frame.iloc[fit_rows]
    docs = fit_x["signal_stream"].fillna("").astype(str).str.strip().tolist()
    analyzer = TfidfVectorizer(**dict(CHAR_TFIDF_PARAMETERS)).build_analyzer()
    unique_ngrams = sum(len(set(analyzer(doc))) for doc in docs)
    cap = int(CHAR_TFIDF_PARAMETERS["max_features"])
    char_nnz = min(unique_ngrams, len(fit_rows) * cap)
    structured_nnz = len(fit_rows) * int(STRUCTURED_NNZ_UPPER_PER_ROW)
    total_nnz = char_nnz + structured_nnz + len(fit_rows)
    csr_bytes = total_nnz * 12 + (len(fit_rows) + 1) * 8
    width = min(cap, unique_ngrams) + 807 + 1
    hist_bytes = width * 256 * 16 * R06_THREADS * 2
    estimate = 3 * csr_bytes + hist_bytes + int(LIGHTGBM_MEMORY_OVERHEAD_BYTES)
    allowed = min(int(JOB_MEMORY_CAP_BYTES), max(0, int(available) - int(FREE_MEMORY_RESERVE_BYTES)))
    return {"status": "PASS" if available >= 4 * 1024**3 and estimate <= allowed else "BLOCKED_RESOURCE",
            "fit_rows": len(fit_rows), "estimated_peak_bytes": int(estimate),
            "allowed_bytes": int(allowed), "available_bytes": int(available),
            "free_memory_reserve_bytes": int(FREE_MEMORY_RESERVE_BYTES),
            "job_memory_cap_bytes": int(JOB_MEMORY_CAP_BYTES), "threads": int(R06_THREADS),
            "estimate_is_observed_peak": False}


def _r11_resource_probe(frame: pd.DataFrame, cycles: np.ndarray, cutoff: int,
                        available: int) -> dict[str, Any]:
    """Fit temporary TF-IDF/SVD on the prefix only to bound R11 memory; no labels/estimator."""
    from sklearn.decomposition import TruncatedSVD
    from src.models.optimization_v3_batch1_r11 import SVD_PARAMETERS, _text_documents
    from src.models.regression_research_text import RegressionTextModel
    from scripts.optimization_v3_batch1_r11 import JOB_CAP_GIB, MIN_FREE_GIB, RESERVE_GIB

    fit_rows = np.flatnonzero(cycles <= cutoff)
    score_rows = _expected_rows(cycles, cutoff + 1, cutoff + 4)
    fit_docs, _ = _text_documents(frame.iloc[fit_rows]["signal_stream"])
    score_docs, _ = _text_documents(frame.iloc[score_rows]["signal_stream"])
    vectorizer, fit_sparse, fallback = RegressionTextModel._fit_vectorizer(fit_docs)
    if vectorizer is None or fit_sparse.shape[1] <= 32:
        return {"status": "BLOCKED_SUPPORT", "fit_rows": len(fit_rows), "score_rows": len(score_rows),
                "predictor_fit_count": 0, "preprocessing_probe_calls": {"tfidf": 1, "svd": 0}}
    score_sparse = vectorizer.transform(score_docs).tocsr()
    svd = TruncatedSVD(n_components=32, **dict(SVD_PARAMETERS))
    fit_projection = svd.fit_transform(fit_sparse)
    score_projection = svd.transform(score_sparse)
    available_gib = available / 1024**3
    allowed_gib = min(float(JOB_CAP_GIB), available_gib - float(RESERVE_GIB))
    sparse_bytes = int(fit_sparse.data.nbytes + fit_sparse.indices.nbytes + fit_sparse.indptr.nbytes
                       + score_sparse.data.nbytes + score_sparse.indices.nbytes + score_sparse.indptr.nbytes)
    projection_bytes = int(fit_projection.nbytes + score_projection.nbytes)
    design_bytes = int((len(fit_rows) + len(score_rows)) * (48 + 32 + 1) * 8)
    estimate = 1.5 + (2 * sparse_bytes + 2 * projection_bytes + 2 * design_bytes) / 1024**3
    return {
        "status": "PASS" if available_gib >= float(MIN_FREE_GIB) and estimate <= allowed_gib else "BLOCKED_RESOURCE",
        "fit_rows": len(fit_rows), "score_rows": len(score_rows), "tfidf_features": int(fit_sparse.shape[1]),
        "sparse_bytes_observed": sparse_bytes, "projection_bytes_observed": projection_bytes,
        "estimated_peak_gib": float(estimate), "allowed_gib": float(allowed_gib),
        "available_gib": float(available_gib), "text_fallback": fallback,
        "preprocessing_probe_calls": {"tfidf": 1, "svd": 1},
        "predictor_fit_count": 0, "validation_labels_used": False, "test_data_used": False,
        "estimate_is_observed_peak": False,
    }


def _build_frozen_preflight(*, write: bool) -> dict[str, Any]:
    if write:
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"], text=True,
        )
        if status.strip():
            raise ExpandedCampaignError("Commit the reviewed continuation source before writing the freeze")
    if not LEDGER_PATH.is_file() or not TRAINING_LOCK.is_file():
        raise ExpandedCampaignError("Active global ledger or preserved one-byte training lock is missing")
    if TRAINING_LOCK.stat().st_size != 1:
        raise ExpandedCampaignError("Shared V3 training lock file identity changed; do not replace it")
    from scripts import optimization_v3_final_continuation_cls as cls_cont
    from scripts import optimization_v3_final_continuation_r22 as reg_cont

    cls_context, reg_context = _load_contexts()
    artifacts = _active_card_artifacts()
    cycles_cls = np.asarray(cls_context["cycles"], dtype=np.int64)
    cycles_reg = np.asarray(reg_context["cycles"], dtype=np.int64)
    pilot_inventory: dict[str, Any] = {}
    for card in NEW_CANDIDATES:
        pilot_inventory[card] = {}
        for cutoff in PILOT_CUTOFFS:
            if card in {"C17", "C11"}:
                controls = cls_context["controls"][cutoff]
                context_frame, context_cycles = cls_context["frame"], cycles_cls
            else:
                source = reg_context["pilots"][cutoff]
                rows = np.asarray(source["rows"], dtype=np.int64)
                controls = {"baseline_prediction": np.asarray(source["baseline_prediction"], dtype=np.float64),
                            "cycle_code": cycles_reg[rows], "control_rows": rows}
                context_frame, context_cycles = reg_context["frame"], cycles_reg
            record = _check_pilot_bundle(card, cutoff, context_frame, context_cycles,
                                         controls, artifacts)
            pilot_inventory[card][str(cutoff)] = {
                "npz_sha256": record["bundle_sha256"],
                "sidecar_sha256": record["sidecar_sha256"],
                "freeze_signature": artifacts[card]["freeze"]["signature"],
                "fit_rows": int(record["metadata"]["fit_rows"]),
                "score_rows": int(len(record["score_rows"])),
                "predictor_fit_count": int(record["metadata"].get("fit_predictor_count", 0)),
            }

    budget = _ledger_budget(expected_reserved=102)
    if budget["reserved_predictor_fits"] + NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS > GLOBAL_CAP:
        raise ExpandedCampaignError("Expanded continuation + full-Train fit plan exceeds the shared cap")
    projected = (budget["reserved_predictor_fits"] + PREVIOUS_NEW_FULL_TRAIN_FITS
                 + NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS)
    if projected != 169 or GLOBAL_CAP - projected != RETRY_HEADROOM:
        raise ExpandedCampaignError("Expanded predictor-fit reconciliation does not match the approved 169/180 plan")
    reuse = _submission_reuse_audit()
    resources = _candidate_resource_estimates(cls_context, reg_context)
    try:
        with exclusive_training_slot(TRAINING_LOCK):
            lock_probe = {"status": "PASS", "path": TRAINING_LOCK.relative_to(ROOT).as_posix(),
                          "process_owned_nonblocking_lock": True}
    except Exception as exc:
        lock_probe = {"status": "BLOCKED_LOCK_BUSY", "path": TRAINING_LOCK.relative_to(ROOT).as_posix(),
                      "error_type": type(exc).__name__, "error": str(exc)[:500]}
    source_hashes, input_hashes, source_drift = _source_and_input_inventory(
        artifacts, cls_context, reg_context)
    old_c16 = _read_json(ROOT / "reports/optimization_v3/final_continuation/C16/results.json")
    old_r22 = _read_json(ROOT / "reports/optimization_v3/final_continuation/R22/result.json")
    if old_c16.get("qualification", {}).get("qualified") is not False:
        raise ExpandedCampaignError("C16 historical gate-failed verdict unexpectedly changed")
    if (old_r22.get("qualification", {}).get("R22_ORIGINAL", {}).get("qualified") is not True
            or old_r22.get("selected_policy") != "R22_ORIGINAL"):
        raise ExpandedCampaignError("R22_ORIGINAL historical qualified verdict unexpectedly changed")

    screening = _read_json(ROOT / "reports/optimization_v3/batch1/batch_ledger.json")
    if int(screening.get("pilot_screening", {}).get("advance", -1)) != 7:
        raise ExpandedCampaignError("The authoritative V3 screen no longer records seven ADVANCE candidates")
    decision_path = ROOT / "reports/optimization_v3/final_continuation_plan/DECISION.md"
    summary_path = ROOT / "reports/optimization_v3/final_continuation/CAMPAIGN_SUMMARY.md"

    gate_protocol = {
        "evaluation": "six disjoint primary windows, H12 from pilot cutoffs 55/67, W5+W6 separately; development labels were previously viewed",
        "primary_rows": EXPECTED_PRIMARY_ROWS,
        "H12_rows": EXPECTED_H12_ROWS,
        "late_rows": EXPECTED_LATE_ROWS,
        "classification": {
            "primary_gain_min": 0.0005, "H12_harm_max": 0.001,
            "minimum_nonnegative_windows": 4, "late_W5_W6_gain_min_exclusive": 0.0,
            "subgroups": ["entry_friction", "stream_availability", "fit-prefix text_load quantiles"],
            "support": "n>=200 and both classes >=30", "macro_f1_harm_max": 0.01,
            "positive_recall_harm_max": 0.02,
        },
        "regression": {
            "primary_gain_min": 0.03, "H12_harm_max": 0.05,
            "minimum_nonnegative_windows": 4, "late_W5_W6_gain_min_exclusive": 0.0,
            "subgroups": ["signal_stream_missing", "threat_rank<=5", "threat_rank>=95"],
            "support_minimum": 100, "MAE_harm_max": 0.25,
        },
        "historical_verdicts_immutable": True,
        "performance_failures_label": "EXPERIMENTAL_GATE_EXCEPTION",
        "qualification_pass_label": "QUALIFIED",
    }
    fit_recipe = {
        "C17": {"continuation_predictors": 4, "full_train_predictors": 1,
                "preprocessing_fit_components": ["structured/OHE", "word TF-IDF"],
                "prediction": ".75 C2_B + .25 C17 for present; frozen CB_MISSING for missing; threshold 0.50"},
        "C11": {"continuation_predictors": 4, "full_train_predictors": 1,
                "preprocessing_fit_components": ["structured/OHE", "word TF-IDF", "character TF-IDF"],
                "prediction": ".75 C2_B + .25 C11 for present; frozen CB_MISSING for missing; threshold 0.50"},
        "R21": {"continuation_predictors": 4, "full_train_predictors": 1,
                "preprocessing_fit_components": ["structured/OHE", "TF-IDF", "SVD32"],
                "prediction": "original reg:absoluteerror; rank50 raw once within complete cycle; .75 champion + .25 mapped candidate; clip [0,100]"},
        "R11": {"continuation_predictors": 36, "full_train_predictors": 9,
                "preprocessing_fit_components": ["structured/OHE", "TF-IDF", "SVD32"],
                "prediction": "nine strict cumulative heads >10..90; PAVA; interpolated median; rank50 once per cycle; .75 champion + .25; clip [0,100]"},
        "R06": {"continuation_predictors": 4, "full_train_predictors": 1,
                "preprocessing_fit_components": ["structured/OHE", "character TF-IDF"],
                "prediction": "original all-row rank50 R06; .75 three-component champion + .25 mapped candidate; clip [0,100]"},
    }
    identity = {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "scope": {"all_authorized_candidates": list(ALL_CANDIDATES),
                  "new_continuation_candidates": list(NEW_CANDIDATES),
                  "historical_candidates_reused": ["C16", "R22_ORIGINAL"],
                  "no_other_recipe_or_threshold": True},
        "source_head_before_freeze": _head(),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "source_drift_audits": source_drift,
        "historical_decision_sha256": file_sha256(decision_path),
        "historical_campaign_summary_sha256": file_sha256(summary_path),
        "screening_authoritative_sha256": file_sha256(ROOT / "reports/optimization_v3/batch1/batch_ledger.json"),
        "official_train": {
            "classification_sha256": file_sha256(cls_context["train_path"]),
            "regression_sha256": file_sha256(reg_context["train_path"]),
            "rows": 50_608, "cycle_min": 47, "cycle_max": 79,
            "test_or_sample_submission_opened": False,
        },
        "active_candidate_recipes": {
            card: {"recipe_sha256": artifacts[card]["recipe_sha256"],
                   "freeze_sha256": artifacts[card]["freeze_sha256"],
                   "freeze_signature": artifacts[card]["freeze"]["signature"],
                   "recipe": _json_clean(artifacts[card]["recipe"])}
            for card in NEW_CANDIDATES
        },
        "reused_pilot_bundles": pilot_inventory,
        "existing_completed_continuation": {
            "C16": {"results_sha256": file_sha256(ROOT / "reports/optimization_v3/final_continuation/C16/results.json"),
                    "verification_sha256": file_sha256(ROOT / "reports/optimization_v3/final_continuation/C16/verification.json"),
                    "qualification": "EXPERIMENTAL_GATE_EXCEPTION"},
            "R22_ORIGINAL": {"result_sha256": file_sha256(ROOT / "reports/optimization_v3/final_continuation/R22/result.json"),
                             "verification_sha256": file_sha256(ROOT / "reports/optimization_v3/final_continuation/R22/verification.json"),
                             "qualification": "QUALIFIED"},
        },
        "full_train_existing_reuse": reuse,
        "continuation_cutoffs": list(NEW_CUTOFFS),
        "reused_pilot_cutoffs": list(PILOT_CUTOFFS),
        "execution_order": [{"candidate": card, "cutoffs": list(NEW_CUTOFFS)} for card in FIT_ORDER],
        "candidate_recipes_and_prediction_policies": fit_recipe,
        "evaluation_protocol": gate_protocol,
        "fit_budget": {
            "cap": GLOBAL_CAP, "prior_reserved_and_saved": budget["reserved_predictor_fits"],
            "prior_reserved_pilot_and_C16_R22_continuation": {"pilot": 94, "continuation": 8},
            "already_completed_offledger_full_train_predictor_fits": PREVIOUS_NEW_FULL_TRAIN_FITS,
            "new_continuation_predictor_fits": NEW_CONTINUATION_FITS,
            "new_full_train_predictor_fits": NEW_FULL_TRAIN_FITS,
            "projected_total_if_complete": projected,
            "retry_headroom_after_complete": RETRY_HEADROOM,
            "old_failed_predictor_attempts": budget["failed_attempts_and_interruptions"],
            "new_control_predictor_fits": 0,
            "preprocessing_and_control_reuse_counted_separately": True,
            "reservation_required_before_first_fit": NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS,
        },
        "resources": {"single_training_job": True, "threads": THREADS,
                      "job_memory_cap_bytes": JOB_CAP_BYTES,
                      "minimum_free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                      "shared_lock_path": TRAINING_LOCK.relative_to(ROOT).as_posix(),
                      "global_lock_probe": lock_probe},
        "kaggle": {"api_auth_available_at_preflight": False, "uploads_allowed_per_candidate": 1,
                   "no_chrome": True, "no_credential_search": True,
                   "live_rules_quota_global_ledger_unavailable": True},
        "test_data_used": False, "sample_submission_used": False,
        "public_score_used_for_recipe_selection": False,
        "package_versions": {name: importlib.metadata.version(name) for name in
                              ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "threadpoolctl", "psutil")},
    }
    signature = json_sha256(identity)
    resource_status = lock_probe.get("status") == "PASS" and all(
        item.get("status") == "PASS"
        for card in resources["by_candidate_cutoff"].values()
        for item in card.values()
    )
    preflight = {
        "schema_version": 1, "status": "READY_TO_FREEZE" if resource_status else "FROZEN_RESOURCE_BLOCKED",
        "created_at_utc": _utc_now(), "freeze_signature": signature,
        "resource_preflight": resources, "fit_budget_snapshot": budget,
        "global_training_lock": lock_probe,
        "full_train_reuse_audit": reuse,
        "pilot_bundles_verified": pilot_inventory,
        "no_new_continuation_scores_opened": True,
        "no_test_or_sample_submission_access": True,
        "proposed_reservation": {"budget_key": NEW_RESERVATION_KEY,
                                 "card_id": NEW_RESERVATION_CARD,
                                 "predictor_fits": NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS,
                                 "reserved": False},
    }
    result = {"identity": _json_clean(identity), "signature": signature,
              "preflight": _json_clean(preflight), "resources_pass": resource_status}
    if write:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        freeze = {"status": "FROZEN_BEFORE_NEW_CONTINUATION",
                  "created_at_utc": _utc_now(), "identity": result["identity"],
                  "signature": signature}
        if FREEZE_PATH.exists():
            prior = _read_json(FREEZE_PATH)
            if prior.get("identity") != result["identity"] or prior.get("signature") != signature:
                raise ExpandedCampaignError("Existing expanded campaign freeze differs; refusing overwrite")
        else:
            write_json_exclusive(FREEZE_PATH, freeze)
        if PREFLIGHT_PATH.exists():
            prior = _read_json(PREFLIGHT_PATH)
            if prior.get("freeze_signature") != signature:
                raise ExpandedCampaignError("Existing expanded preflight is bound to another freeze")
        else:
            write_json_exclusive(PREFLIGHT_PATH, result["preflight"])
    return result


def _verify_frozen_preflight() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not FREEZE_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise ExpandedCampaignError("Expanded campaign freeze/preflight has not been committed")
    frozen = _read_json(FREEZE_PATH)
    preflight = _read_json(PREFLIGHT_PATH)
    identity = frozen.get("identity", {})
    signature = json_sha256(identity)
    if frozen.get("signature") != signature or preflight.get("freeze_signature") != signature:
        raise ExpandedCampaignError("Expanded campaign freeze signature or preflight binding is invalid")
    if frozen.get("status") != "FROZEN_BEFORE_NEW_CONTINUATION":
        raise ExpandedCampaignError("Expanded campaign is not in the frozen-before-fit state")
    if preflight.get("status") != "READY_TO_FREEZE":
        raise ExpandedCampaignError("Preflight recorded a resource or data blocker")
    cls_context, reg_context = _load_contexts()
    artifacts = _active_card_artifacts()
    sources, inputs, _ = _source_and_input_inventory(artifacts, cls_context, reg_context)
    if sources != identity.get("source_sha256") or inputs != identity.get("input_sha256"):
        raise ExpandedCampaignError("Source/input hashes differ from the committed expanded freeze")
    if int(identity.get("source_head_before_freeze", "0"), 16) == 0:
        raise ExpandedCampaignError("Frozen source HEAD is invalid")
    current_ledger = _ledger_budget()
    expected_reservation = current_ledger["reservation_keys"].get(NEW_RESERVATION_KEY)
    if current_ledger["reserved_predictor_fits"] == 102:
        if expected_reservation is not None:
            raise ExpandedCampaignError("Expanded reservation exists but the shared ledger total is still 102")
    elif current_ledger["reserved_predictor_fits"] == 167:
        if expected_reservation != [NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS, NEW_RESERVATION_CARD]:
            raise ExpandedCampaignError("Expanded global fit reservation differs from frozen plan")
    else:
        raise ExpandedCampaignError(
            f"Shared V3 fit ledger changed after freeze: {current_ledger['reserved_predictor_fits']} reserved"
        )
    return identity, cls_context, reg_context


def _controls_for(card: str, cutoff: int,
                  cls_context: Mapping[str, Any], reg_context: Mapping[str, Any]) -> dict[str, Any]:
    if card in {"C17", "C11"}:
        raw = cls_context["controls"][cutoff]
        controls = {k: dict(v) if isinstance(v, Mapping) else np.asarray(v)
                    for k, v in raw.items()}
        controls["control_rows"] = np.asarray(controls["control_rows"], dtype=np.int64)
        return controls
    raw = reg_context["primary_controls"][cutoff]
    controls = {k: dict(v) if isinstance(v, Mapping) else np.asarray(v)
                for k, v in raw.items()}
    controls["control_rows"] = np.asarray(controls["rows"], dtype=np.int64)
    controls["cycle_code"] = np.asarray(controls["cycle_codes"], dtype=np.int64)
    if card == "R21":
        required = {"baseline_prediction", "baseline_lgb_svd32", "baseline_hybrid_ridge",
                    "baseline_sparse_raw", "baseline_sparse_rank50", "control_rows"}
        absent = sorted(required - set(controls))
        if absent:
            raise ExpandedCampaignError(f"R21 frozen champion controls are missing {absent}")
    return controls


def _record_identity(card: str, cutoff: int, rows: np.ndarray,
                     freeze_signature: str) -> dict[str, Any]:
    return {"campaign_id": CAMPAIGN_ID, "campaign_freeze_signature": freeze_signature,
            "card_id": card, "cutoff": int(cutoff), "score_rows": np.asarray(rows, dtype=np.int64).tolist()}


def _new_bundle_path(card: str, cutoff: int) -> Path:
    return RUN_DIR / card / f"{card}_fit{int(cutoff)}.npz"


def _candidate_reconstruct(card: str, raw: np.ndarray, baseline: np.ndarray,
                           cycles: np.ndarray, controls: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if card in {"C17", "C11"}:
        missing = np.asarray(controls["stream_missing"], dtype=bool)
        c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
        cb = np.asarray(controls["cb_probability"], dtype=np.float64)
        return np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    return np.clip(0.75 * baseline + 0.25 * _rank50_independent(raw, cycles), 0.0, 100.0)


def _load_new_bundle(card: str, cutoff: int, frame: pd.DataFrame,
                     cycles: np.ndarray, controls: Mapping[str, Any],
                     freeze_signature: str) -> dict[str, Any]:
    path = _new_bundle_path(card, cutoff)
    expected_rows = _expected_rows(cycles, cutoff + 1, cutoff + 4)
    identity = _record_identity(card, cutoff, expected_rows, freeze_signature)
    arrays, sidecar = read_bundle(path, identity)
    rows = np.asarray(arrays.get("score_rows"), dtype=np.int64)
    ids = np.asarray(arrays.get("score_record_ids"), dtype=str)
    if (not np.array_equal(rows, expected_rows)
            or not np.array_equal(ids, frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str))
            or not np.array_equal(rows, np.asarray(controls["control_rows"], dtype=np.int64))):
        raise ExpandedCampaignError(f"{card} cutoff {cutoff} output alignment failed")
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    prediction = np.asarray(arrays["prediction"], dtype=np.float64)
    baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    saved_cycles = np.asarray(arrays["score_cycles"], dtype=np.int64)
    if (not np.array_equal(saved_cycles, cycles[rows]) or raw.shape != rows.shape
            or prediction.shape != rows.shape or baseline.shape != rows.shape
            or not all(np.isfinite(x).all() for x in (raw, prediction, baseline))):
        raise ExpandedCampaignError(f"{card} cutoff {cutoff} output values are invalid")
    reconstructed = _candidate_reconstruct(card, raw, baseline, saved_cycles, controls)
    if not np.allclose(prediction, reconstructed, atol=1e-10, rtol=0):
        raise ExpandedCampaignError(f"{card} cutoff {cutoff} independent recipe reconstruction failed")
    metadata = sidecar.get("metadata", {})
    expected_mappings = {name: _json_clean(value) for name, value in controls.items()
                         if isinstance(value, Mapping)}
    if metadata.get("control_mappings", {}) != expected_mappings:
        raise ExpandedCampaignError(f"{card} cutoff {cutoff} control mapping audit differs")
    for name, value in controls.items():
        if isinstance(value, Mapping):
            continue
        field = f"control__{name}"
        if field not in arrays or not np.array_equal(np.asarray(arrays[field]), np.asarray(value)):
            raise ExpandedCampaignError(f"{card} cutoff {cutoff} saved control array differs: {name}")
    state = metadata.get("state", {})
    audited_model_state = state.get("model", state) if isinstance(state, Mapping) else {}
    if (metadata.get("fit_predictor_count") != (9 if card == "R11" else 1)
            or metadata.get("test_data_used") is not False
            or metadata.get("validation_labels_passed_to_fit") is not False
            or any(audited_model_state.get(key) is not False
                   for key in ("test_data_used", "validation_labels_passed_to_fit")
                   if key in audited_model_state)):
        raise ExpandedCampaignError(f"{card} cutoff {cutoff} fit metadata is inconsistent")
    return {"score_rows": rows, "score_record_ids": ids, "score_cycles": saved_cycles,
            "prediction": prediction, "raw_candidate": raw, "baseline_prediction": baseline,
            "control_arrays": dict(controls), "metadata": metadata, "bundle_identity": identity,
            "bundle_sha256": file_sha256(path), "sidecar_sha256": file_sha256(path.with_suffix(".json"))}


def _runtime_resource_check(card: str, cutoff: int, estimate: Mapping[str, Any]) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_CAP_BYTES, max(0, available - FREE_RAM_RESERVE_BYTES))
    estimated = estimate.get("estimated_peak_bytes")
    if estimated is None and estimate.get("estimated_peak_gib") is not None:
        estimated = int(float(estimate["estimated_peak_gib"]) * 1024**3)
    if estimated is None:
        return {"status": "BLOCKED_UNKNOWN_RESOURCE", "available_bytes": available,
                "allowed_bytes": allowed, "estimate": None}
    status = "PASS" if available >= 4 * 1024**3 and int(estimated) <= allowed else "BLOCKED_RESOURCE"
    return {"status": status, "card_id": card, "cutoff": int(cutoff),
            "available_bytes": available, "allowed_bytes": allowed,
            "estimated_peak_bytes": int(estimated), "job_cap_bytes": JOB_CAP_BYTES,
            "free_reserve_bytes": FREE_RAM_RESERVE_BYTES, "threads": THREADS,
            "single_training_job": True, "estimate_is_observed_peak": False}


def _run_one_fit(card: str, cutoff: int, cls_context: Mapping[str, Any],
                 reg_context: Mapping[str, Any], freeze_signature: str,
                 attempt_number: int) -> dict[str, Any]:
    is_classification = card in {"C17", "C11"}
    context = cls_context if is_classification else reg_context
    frame = context["frame"]
    target = np.asarray(context["target"])
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    features = context["features"] if "features" in context else None
    if features is None:
        feature_names = ("impact_load", "entry_friction", "cycle_code", "scope_pressure", "text_load",
                         "scope_trace", "org_spread", "module_spread", "telemetry_flag", "org_code",
                         "module_code", "pattern_code", "signal_stream")
        features = frame.loc[:, list(feature_names)]
    fit_rows = np.flatnonzero(cycles <= cutoff).astype(np.int64)
    score_rows = _expected_rows(cycles, cutoff + 1, cutoff + 4)
    controls = _controls_for(card, cutoff, cls_context, reg_context)
    if not np.array_equal(np.asarray(controls["control_rows"], dtype=np.int64), score_rows):
        raise ExpandedCampaignError(f"{card} cutoff {cutoff} controls miss the exact score rows")
    fit_x = features.iloc[fit_rows]
    score_x = features.iloc[score_rows]
    fit_y = target[fit_rows]
    estimate = _read_json(PREFLIGHT_PATH)["resource_preflight"]["by_candidate_cutoff"][card][str(cutoff)]
    resource = _runtime_resource_check(card, cutoff, estimate)
    if resource["status"] != "PASS":
        raise ExpandedCampaignError(f"Resource preflight blocks {card}/{cutoff}: {resource}")
    expected_fits = 9 if card == "R11" else 1
    path = _new_bundle_path(card, cutoff)
    if path.exists() or path.with_suffix(".json").exists():
        raise ExpandedCampaignError(f"Refusing to overwrite or complete partial bundle {path}")

    attempt_event = {
        "event": "EXPANDED_CONTINUATION_FIT_STARTED", "at_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID, "freeze_signature": freeze_signature,
        "card_id": card, "cutoff": cutoff, "attempt": attempt_number,
        "reserved_predictor_fits": expected_fits,
        "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
        "resource_preflight": resource,
    }
    append_event(LEDGER_PATH, attempt_event)
    attempts_path = REPORT_DIR / card / "attempts.jsonl"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_path.open("a", encoding="utf-8", newline="\n").write(
        json.dumps(_json_clean(attempt_event), sort_keys=True, allow_nan=False) + "\n")
    started = time.monotonic()
    result = FIT_FUNCTIONS[card](fit_x, fit_y, score_x, controls=controls, cutoff=cutoff)
    elapsed = time.monotonic() - started
    if int(result["fit_predictor_count"]) != expected_fits:
        raise ExpandedCampaignError(f"{card}/{cutoff} reported {result['fit_predictor_count']} fits; expected {expected_fits}")
    baseline = np.asarray(result["baseline_prediction"], dtype=np.float64)
    raw = np.asarray(result["raw_candidate"], dtype=np.float64)
    prediction = np.asarray(result["prediction"], dtype=np.float64)
    if not np.array_equal(baseline, np.asarray(controls["baseline_prediction"], dtype=np.float64)):
        raise ExpandedCampaignError(f"{card}/{cutoff} changed the frozen comparison baseline")
    reconstructed = _candidate_reconstruct(card, raw, baseline, cycles[score_rows], controls)
    if not np.allclose(prediction, reconstructed, atol=1e-10, rtol=0):
        raise ExpandedCampaignError(f"{card}/{cutoff} adapter result does not independently reconstruct")
    expected_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
    identity = _record_identity(card, cutoff, score_rows, freeze_signature)
    arrays: dict[str, np.ndarray] = {
        "score_rows": score_rows,
        "score_record_ids": expected_ids,
        "score_cycles": cycles[score_rows],
        "raw_candidate": raw,
        "prediction": prediction,
        "baseline_prediction": baseline,
    }
    for name, value in result.get("audit_arrays", {}).items():
        arrays[f"audit__{name}"] = np.asarray(value)
    control_mappings: dict[str, Any] = {}
    for name, value in controls.items():
        if isinstance(value, Mapping):
            control_mappings[name] = _json_clean(value)
        else:
            arrays[f"control__{name}"] = np.asarray(value)
    metadata = {
        "campaign_id": CAMPAIGN_ID, "card_id": card, "cutoff": int(cutoff),
        "fit_predictor_count": expected_fits, "fit_seconds": float(elapsed),
        "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
        "resource_preflight": resource, "fit_metadata": _json_clean(result["fit_metadata"]),
        "state": _json_clean(result["state"]), "freeze_signature": freeze_signature,
        "control_mappings": control_mappings,
        "validation_labels_passed_to_fit": False, "test_data_used": False,
        "preprocessing_fit_components": list(
            {"C17": ("structured/OHE", "word TF-IDF"),
             "C11": ("structured/OHE", "word TF-IDF", "character TF-IDF"),
             "R21": ("structured/OHE", "TF-IDF", "SVD32"),
             "R11": ("structured/OHE", "TF-IDF", "SVD32"),
             "R06": ("structured/OHE", "character TF-IDF")}[card]),
    }
    write_bundle_exclusive(path, arrays, identity, metadata)
    saved = _load_new_bundle(card, cutoff, frame, cycles, controls, freeze_signature)
    saved_event = {
        "event": "EXPANDED_CONTINUATION_FIT_SAVED", "at_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID, "freeze_signature": freeze_signature,
        "card_id": card, "cutoff": cutoff, "attempt": attempt_number,
        "predictor_fits_saved": expected_fits, "fit_seconds": float(elapsed),
        "bundle_sha256": saved["bundle_sha256"], "sidecar_sha256": saved["sidecar_sha256"],
        "state_sha256": metadata["state"].get("state_sha256"),
    }
    append_event(LEDGER_PATH, saved_event)
    attempts_path.open("a", encoding="utf-8", newline="\n").write(
        json.dumps(_json_clean(saved_event), sort_keys=True, allow_nan=False) + "\n")
    return saved_event


def run_continuations() -> dict[str, Any]:
    identity, cls_context, reg_context = _verify_frozen_preflight()
    with exclusive_training_slot(TRAINING_LOCK):
        return _run_continuations_locked(identity, cls_context, reg_context)


def _run_continuations_locked(identity: Mapping[str, Any], cls_context: Mapping[str, Any],
                              reg_context: Mapping[str, Any]) -> dict[str, Any]:
    freeze_signature = json_sha256(identity)
    budget_before = _ledger_budget()
    existing = budget_before["reservation_keys"].get(NEW_RESERVATION_KEY)
    expected_value = [NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS, NEW_RESERVATION_CARD]
    if budget_before["reserved_predictor_fits"] == 102 and existing is None:
        reserve = reserve_card_budget(
            LEDGER_PATH, budget_key=NEW_RESERVATION_KEY, card_id=NEW_RESERVATION_CARD,
            predictor_fits=NEW_CONTINUATION_FITS + NEW_FULL_TRAIN_FITS, cap=GLOBAL_CAP,
        )
    elif budget_before["reserved_predictor_fits"] == 167 and existing == expected_value:
        reserve = {"status": "IDEMPOTENT_EXISTING_RESERVATION", "predictor_fits": expected_value[0],
                   "card_id": expected_value[1], "budget_key": NEW_RESERVATION_KEY}
    else:
        raise ExpandedCampaignError(f"Expanded reservation/ledger is ambiguous before continuation: {budget_before}")
    after_reserve = _ledger_budget(expected_reserved=167)
    if after_reserve["reservation_keys"].get(NEW_RESERVATION_KEY) != expected_value:
        raise ExpandedCampaignError("Expanded global fit reservation differs from frozen plan")
    if after_reserve["reserved_predictor_fits"] + PREVIOUS_NEW_FULL_TRAIN_FITS > GLOBAL_CAP:
        raise ExpandedCampaignError("Including prior full-Train fits, projected budget exceeds 180")
    if not LEDGER_PATH.is_file() or not TRAINING_LOCK.is_file() or TRAINING_LOCK.stat().st_size != 1:
        raise ExpandedCampaignError("Shared lock/ledger changed before continuation execution")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    start_event = {"event": "EXPANDED_CONTINUATION_EXECUTION_STARTED", "at_utc": _utc_now(),
                   "campaign_id": CAMPAIGN_ID, "freeze_signature": freeze_signature,
                "budget_before": budget_before, "budget_after_reservation": after_reserve,
                   "reservation": reserve}
    append_event(LEDGER_PATH, start_event)
    with nullcontext():
        for card in FIT_ORDER:
            for cutoff in NEW_CUTOFFS:
                frame = cls_context["frame"] if card in {"C17", "C11"} else reg_context["frame"]
                cycles = np.asarray(cls_context["cycles"] if card in {"C17", "C11"}
                                    else reg_context["cycles"], dtype=np.int64)
                controls = _controls_for(card, cutoff, cls_context, reg_context)
                path = _new_bundle_path(card, cutoff)
                try:
                    if path.is_file() and path.with_suffix(".json").is_file():
                        record = _load_new_bundle(card, cutoff, frame, cycles, controls, freeze_signature)
                        receipt = {"status": "VERIFIED_REUSE", "card_id": card, "cutoff": cutoff,
                                   "predictor_fits_saved": 0,
                                   "bundle_sha256": record["bundle_sha256"],
                                   "sidecar_sha256": record["sidecar_sha256"]}
                    else:
                        attempt_dir = REPORT_DIR / card
                        attempt_path = attempt_dir / "attempts.jsonl"
                        previous = []
                        if attempt_path.is_file():
                            previous = [json.loads(line) for line in attempt_path.read_text(encoding="utf-8").splitlines()
                                        if line.strip()]
                        started_count = sum(e.get("event") == "EXPANDED_CONTINUATION_FIT_STARTED"
                                            and int(e.get("cutoff", -1)) == cutoff for e in previous)
                        if started_count:
                            raise ExpandedCampaignError(
                                f"Prior started fit for {card}/{cutoff} has no valid bundle; manual integrity reconciliation required"
                            )
                        receipt = _run_one_fit(card, cutoff, cls_context, reg_context,
                                               freeze_signature, attempt_number=1)
                        receipt["status"] = "FIT_SAVED"
                    receipts.append(receipt)
                except Exception as exc:
                    error = {"status": "FIT_OR_INTEGRITY_BLOCKED", "card_id": card,
                             "cutoff": cutoff, "error_type": type(exc).__name__,
                             "error": str(exc)[:1600], "at_utc": _utc_now()}
                    errors.append(error)
                    append_event(LEDGER_PATH, {"event": "EXPANDED_CONTINUATION_FIT_FAILED",
                                               "campaign_id": CAMPAIGN_ID,
                                               "freeze_signature": freeze_signature, **error})
                    REPORT_DIR.mkdir(parents=True, exist_ok=True)
                    attempt_path = REPORT_DIR / card / "attempts.jsonl"
                    attempt_path.parent.mkdir(parents=True, exist_ok=True)
                    attempt_path.open("a", encoding="utf-8", newline="\n").write(
                        json.dumps(error, sort_keys=True, allow_nan=False) + "\n")
                    # Candidate independence permits continuing the fixed campaign order.
                    continue
    finished = {"event": "EXPANDED_CONTINUATION_EXECUTION_FINISHED", "at_utc": _utc_now(),
                "campaign_id": CAMPAIGN_ID, "freeze_signature": freeze_signature,
                "fit_errors": errors, "run_receipts": receipts,
                "budget_after_fits": _ledger_budget(expected_reserved=167)}
    append_event(LEDGER_PATH, finished)
    return {"status": "CONTINUATION_FITS_COMPLETE" if not errors else "CONTINUATION_PARTIAL",
            "freeze_signature": freeze_signature, "reservation": reserve,
            "run_receipts": receipts, "fit_errors": errors,
            "budget_after_fits": finished["budget_after_fits"]}


def _candidate_resource_estimates(cls_context: Mapping[str, Any], reg_context: Mapping[str, Any]) -> dict[str, Any]:
    from scripts import optimization_v3_batch1_c11 as c11_runner
    from scripts import optimization_v3_batch1_c17 as c17_runner
    from scripts import optimization_v3_batch1_r21 as r21_runner
    available = int(psutil.virtual_memory().available)
    output: dict[str, Any] = {"available_ram_bytes_at_preflight": available,
                              "available_ram_gib_at_preflight": available / 1024**3,
                              "single_training_job": True, "threads": THREADS,
                              "job_memory_cap_bytes": JOB_CAP_BYTES,
                              "free_ram_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                              "by_candidate_cutoff": {}}
    for candidate in NEW_CANDIDATES:
        frame = cls_context["frame"] if candidate in {"C17", "C11"} else reg_context["frame"]
        cycles = cls_context["cycles"] if candidate in {"C17", "C11"} else reg_context["cycles"]
        output["by_candidate_cutoff"][candidate] = {}
        for cutoff in NEW_CUTOFFS:
            if candidate == "C17":
                rows = np.flatnonzero(cycles <= cutoff)
                record = c17_runner._memory_estimate(frame.iloc[rows].loc[:, list(c17_runner.FEATURE_NAMES)], available)
            elif candidate == "C11":
                rows = np.flatnonzero(cycles <= cutoff)
                record = c11_runner._resource_estimate(frame.iloc[rows].loc[:, list(c11_runner.FEATURE_NAMES)])
                record["available_physical_bytes"] = available
            elif candidate == "R21":
                record = r21_runner._resource_preflight(int(np.sum(cycles <= cutoff)), available)
            elif candidate == "R06":
                record = _r06_resource_estimate(frame, cycles, cutoff, available)
            else:
                record = _r11_resource_probe(frame, cycles, cutoff, available)
            output["by_candidate_cutoff"][candidate][str(cutoff)] = _json_clean(record)
    return output


def _source_and_input_inventory(card_artifacts: Mapping[str, Mapping[str, Any]],
                                cls_context: Mapping[str, Any], reg_context: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    sources: dict[str, str] = {}
    historical_expected: dict[str, dict[str, str]] = {}
    shared_mismatch_expected = "d21ac914d0d1f516a93e431116556dabdbe7313a21fd2b1b04a73371bd56f420"
    for card, active in card_artifacts.items():
        mapping = active["freeze"].get("identity", {}).get("source_sha256", {})
        historical_expected[card] = {}
        for raw_path, expected in mapping.items():
            path = _source_path(raw_path)
            if not path.is_file():
                raise ExpandedCampaignError(f"Frozen {card} source is missing: {path}")
            observed = file_sha256(path)
            historical_expected[card][path.relative_to(ROOT).as_posix()] = str(expected)
            if observed != expected:
                if card in {"C17", "R21"} and path.relative_to(ROOT).as_posix() == "src/models/optimization_v3_batch1.py" and expected == shared_mismatch_expected:
                    pass
                else:
                    raise ExpandedCampaignError(f"Unexpected source drift in {card}: {path.relative_to(ROOT)}")
            sources[path.relative_to(ROOT).as_posix()] = observed

    for context in (cls_context, reg_context):
        for rel, expected in context["identity"].get("source_sha256", {}).items():
            path = _source_path(rel)
            observed = file_sha256(path)
            if observed != expected:
                # C16 and R22 continuation freezes already carry the audited shared-engine amendment.
                if path.relative_to(ROOT).as_posix() != "src/models/optimization_v3_batch1.py":
                    raise ExpandedCampaignError(f"Existing continuation context source drift: {rel}")
            sources[path.relative_to(ROOT).as_posix()] = observed
    own = [
        Path(__file__).resolve(),
        ROOT / "src/models/optimization_v3_expanded_continuation.py",
        ROOT / "scripts/optimization_v3_final_continuation_cls.py",
        ROOT / "scripts/optimization_v3_final_continuation_r22.py",
        ROOT / "scripts/optimization_v3_batch1_r06.py",
    ]
    for path in own:
        sources[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    source_rebinding = _audit_shared_engine_rebinding(shared_mismatch_expected)
    inputs: dict[str, str] = {}
    for context_name, context in (("classification_controls", cls_context), ("regression_controls", reg_context)):
        for name, digest in context["identity"].get("input_sha256", {}).items():
            inputs[f"{context_name}:{name}"] = str(digest)
    for card, active in card_artifacts.items():
        inputs[f"{card}:recipe"] = active["recipe_sha256"]
        inputs[f"{card}:freeze"] = active["freeze_sha256"]
        for cutoff in PILOT_CUTOFFS:
            bundle = _active_pilot_dir(card) / f"{card}_fit{cutoff}.npz"
            inputs[f"{card}:pilot{cutoff}:npz"] = file_sha256(bundle)
            inputs[f"{card}:pilot{cutoff}:sidecar"] = file_sha256(bundle.with_suffix(".json"))
    for rel in (
        "reports/optimization_v3/final_continuation_plan/DECISION.md",
        "reports/optimization_v3/final_continuation_plan/guard_audit.json",
        "reports/optimization_v3/final_continuation/CAMPAIGN_SUMMARY.md",
        "reports/optimization_v3/final_continuation/C16/results.json",
        "reports/optimization_v3/final_continuation/C16/verification.json",
        "reports/optimization_v3/final_continuation/R22/result.json",
        "reports/optimization_v3/final_continuation/R22/verification.json",
        "reports/submissions/c16_r22/SUBMISSION_STATUS.json",
        "submissions/c16_r22/classification_c16.csv",
        "submissions/c16_r22/regression_r22_original.csv",
    ):
        path = ROOT / rel
        inputs[rel] = file_sha256(path)
    return dict(sorted(sources.items())), dict(sorted(inputs.items())), {
        "historical_source_sha256_by_candidate": historical_expected,
        "shared_engine_rebinding_audit": source_rebinding,
    }


def _records_for_candidate(card: str, cls_context: Mapping[str, Any],
                           reg_context: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]],
                           freeze_signature: str) -> dict[int, dict[str, Any]]:
    is_classification = card in {"C17", "C11"}
    context = cls_context if is_classification else reg_context
    frame = context["frame"]
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    records: dict[int, dict[str, Any]] = {}
    for cutoff in ALL_ORIGINS:
        if cutoff in PILOT_CUTOFFS:
            if is_classification:
                controls = cls_context["controls"][cutoff]
            else:
                pilot = reg_context["pilots"][cutoff]
                rows = np.asarray(pilot["rows"], dtype=np.int64)
                controls = {
                    "baseline_prediction": np.asarray(pilot["baseline_prediction"], dtype=np.float64),
                    "cycle_code": cycles[rows], "control_rows": rows,
                }
            record = _check_pilot_bundle(card, cutoff, frame, cycles, controls, artifacts)
        else:
            controls = _controls_for(card, cutoff, cls_context, reg_context)
            record = _load_new_bundle(card, cutoff, frame, cycles, controls, freeze_signature)
        expected_low, expected_high = ((cutoff + 1, cutoff + 12) if cutoff in PILOT_CUTOFFS
                                       else (cutoff + 1, cutoff + 4))
        expected_rows = _expected_rows(cycles, expected_low, expected_high)
        if not np.array_equal(np.asarray(record["score_rows"], dtype=np.int64), expected_rows):
            raise ExpandedCampaignError(f"{card}/{cutoff} record coverage differs from its frozen horizon")
        record["score_record_ids"] = frame.iloc[expected_rows]["record_id"].astype(str).to_numpy(dtype=str)
        record["origins"] = np.full(len(expected_rows), cutoff, dtype=np.int64)
        records[cutoff] = record
    if set(records) != set(ALL_ORIGINS):
        raise ExpandedCampaignError(f"{card} evaluation requires all six frozen origins")
    return records


def _classification_candidate_report(card: str, frame: pd.DataFrame, target: np.ndarray,
                                     records: Mapping[int, Mapping[str, Any]],
                                     freeze_signature: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts import optimization_v3_final_continuation_cls as cls_report

    declared = cls_report.build_report(frame, target, records, freeze_signature)
    verification = cls_report.verify_report_independently(
        frame, target, records, declared, freeze_signature,
    )
    if verification.get("status") != "PASS":
        raise ExpandedCampaignError(f"{card} independent classification verification failed")
    declared["card_id"] = card
    declared["candidate"] = card
    declared["late_W5_W6"] = declared["qualification"]["late_W5_W6"]
    declared["qualification"]["status"] = (
        "QUALIFIED" if declared["qualification"]["qualified"] else "EXPERIMENTAL_GATE_EXCEPTION"
    )
    verification["card_id"] = card
    verification["candidate"] = card
    return declared, verification


def _regression_views(frame: pd.DataFrame, cycles: np.ndarray,
                      records: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    from scripts import optimization_v3_final_continuation_r22 as reg_report

    views: dict[str, dict[str, np.ndarray]] = {}
    for name, (cutoff, low, high) in reg_report.WINDOWS.items():
        record = records[cutoff]
        rows = np.asarray(record["score_rows"], dtype=np.int64)
        mask = (cycles[rows] >= low) & (cycles[rows] <= high)
        selected = rows[mask]
        expected = _expected_rows(cycles, low, high)
        if not np.array_equal(selected, expected):
            raise ExpandedCampaignError(f"Regression {name} row coverage differs from official Train")
        views[name] = {
            "rows": selected,
            "baseline": np.asarray(record["baseline_prediction"], dtype=np.float64)[mask],
            "candidate": np.asarray(record["prediction"], dtype=np.float64)[mask],
            "origins": np.full(len(selected), cutoff, dtype=np.int64),
        }
    primary_rows = np.concatenate([views[f"W{i}"]["rows"] for i in range(1, 7)])
    expected_primary = _expected_rows(cycles, 56, 79)
    if (len(primary_rows) != EXPECTED_PRIMARY_ROWS
            or len(np.unique(primary_rows)) != EXPECTED_PRIMARY_ROWS
            or not np.array_equal(np.sort(primary_rows), expected_primary)):
        raise ExpandedCampaignError("Regression primary rows are not 38,588 unique rows for cycles 56–79")
    views["primary"] = {
        key: np.concatenate([views[f"W{i}"][key] for i in range(1, 7)])
        for key in ("rows", "baseline", "candidate", "origins")
    }
    early, late = records[55], records[67]
    h12_rows = np.concatenate([np.asarray(early["score_rows"], dtype=np.int64),
                               np.asarray(late["score_rows"], dtype=np.int64)])
    if (len(h12_rows) != EXPECTED_H12_ROWS
            or len(np.unique(h12_rows)) != EXPECTED_H12_ROWS
            or not np.array_equal(np.sort(h12_rows), expected_primary)):
        raise ExpandedCampaignError("Regression H12 pilot outputs do not cover 38,588 unique primary rows")
    h12 = {
        "rows": h12_rows,
        "baseline": np.concatenate([np.asarray(early["baseline_prediction"], dtype=np.float64),
                                    np.asarray(late["baseline_prediction"], dtype=np.float64)]),
        "candidate": np.concatenate([np.asarray(early["prediction"], dtype=np.float64),
                                     np.asarray(late["prediction"], dtype=np.float64)]),
        "origins": np.concatenate([np.full(len(early["score_rows"]), 55, dtype=np.int64),
                                   np.full(len(late["score_rows"]), 67, dtype=np.int64)]),
    }
    views["H12"] = h12
    for name, low, high in (("H12_E", 56, 67), ("H12_L", 68, 79)):
        mask = (cycles[h12_rows] >= low) & (cycles[h12_rows] <= high)
        views[name] = {key: np.asarray(value)[mask] for key, value in h12.items()}
    late_rows = np.concatenate([views["W5"]["rows"], views["W6"]["rows"]])
    if len(np.unique(late_rows)) != EXPECTED_LATE_ROWS:
        raise ExpandedCampaignError("Regression W5+W6 coverage is not 15,953 unique rows")
    return views


def _verify_regression_metrics_independently(frame: pd.DataFrame, target: np.ndarray,
                                             views: Mapping[str, Mapping[str, np.ndarray]],
                                             records: Mapping[int, Mapping[str, Any]],
                                             declared: Mapping[str, Any]) -> dict[str, Any]:
    from scripts import optimization_v3_final_continuation_r22 as reg_report

    y = np.asarray(target, dtype=np.float64)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    missing_all = reg_report.stream_missing_mask(frame["signal_stream"].tolist())
    verified: dict[str, Any] = {"views": {}, "rank_mapping": {}, "group_failures": []}
    for cutoff, record in records.items():
        raw = np.asarray(record["raw_candidate"], dtype=np.float64)
        rows = np.asarray(record["score_rows"], dtype=np.int64)
        score_cycles = cycles[rows]
        mapped = np.empty(raw.shape, dtype=np.float64)
        for cycle in np.unique(score_cycles):
            mask = score_cycles == cycle
            mapped[mask] = 0.5 * raw[mask] + 0.5 * (
                100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
            )
        rebuilt = np.clip(0.75 * np.asarray(record["baseline_prediction"], dtype=np.float64)
                          + 0.25 * mapped, 0.0, 100.0)
        if not np.allclose(rebuilt, np.asarray(record["prediction"], dtype=np.float64), atol=1e-10, rtol=0):
            raise ExpandedCampaignError(f"Independent rank50 recipe reconstruction failed for cutoff {cutoff}")
        verified["rank_mapping"][str(cutoff)] = {
            "status": "PASS", "full_cycle_rows": int(len(rows)),
            "average_ties": True, "mapped_once_before_blend": True,
        }

    metric_rows: dict[str, Any] = {}
    group_failures: list[tuple[str, str]] = []
    for view_name in reg_report.REQUIRED_VIEWS:
        view = views[view_name]
        rows = np.asarray(view["rows"], dtype=np.int64)
        baseline = np.asarray(view["baseline"], dtype=np.float64)
        candidate = np.asarray(view["candidate"], dtype=np.float64)
        if (len(np.unique(rows)) != len(rows)
                or not np.isfinite(baseline).all() or not np.isfinite(candidate).all()
                or np.any((candidate < 0) | (candidate > 100))):
            raise ExpandedCampaignError(f"Independent regression view integrity failed: {view_name}")
        actual = y[rows]
        cand_mae = float(np.mean(np.abs(actual - candidate)))
        base_mae = float(np.mean(np.abs(actual - baseline)))
        saved = declared["views"][view_name]
        if (int(saved["n"]) != len(rows)
                or not np.isclose(cand_mae, float(saved["candidate_mae"]), atol=1e-12, rtol=0)
                or not np.isclose(base_mae, float(saved["champion_mae"]), atol=1e-12, rtol=0)
                or not np.isclose(base_mae - cand_mae, float(saved["gain_vs_champion"]), atol=1e-12, rtol=0)):
            raise ExpandedCampaignError(f"Independent regression metric mismatch: {view_name}")
        metric_rows[view_name] = {"n": int(len(rows)), "candidate_mae": cand_mae,
                                  "champion_mae": base_mae, "gain_vs_champion": base_mae - cand_mae}

        masks = {
            "signal_stream_missing": missing_all[rows],
            "threat_rank_le_5": actual <= 5.0,
            "threat_rank_ge_95": actual >= 95.0,
        }
        for group_name, mask in masks.items():
            n = int(mask.sum())
            support = n >= reg_report.MIN_GROUP_SUPPORT
            if n:
                group_candidate = float(np.mean(np.abs(actual[mask] - candidate[mask])))
                group_baseline = float(np.mean(np.abs(actual[mask] - baseline[mask])))
                harm = group_candidate - group_baseline
            else:
                group_candidate = group_baseline = harm = None
            group_declared = declared["groups"]["views"][view_name][group_name]
            if (n != int(group_declared["n"]) or support != bool(group_declared["supported"])
                    or (harm is not None and not np.isclose(harm, float(group_declared["harm"]), atol=1e-12, rtol=0))):
                raise ExpandedCampaignError(f"Independent subgroup guard mismatch: {view_name}/{group_name}")
            if support and harm > reg_report.MAX_GROUP_HARM:
                group_failures.append((view_name, group_name))
    declared_failures = sorted((item["view"], item["group"])
                               for item in declared["qualification"]["group_failures"])
    if sorted(group_failures) != declared_failures:
        raise ExpandedCampaignError("Independent regression subgroup failure set differs")
    late_rows = np.concatenate([views["W5"]["rows"], views["W6"]["rows"]]).astype(np.int64)
    late_candidate = np.concatenate([views["W5"]["candidate"], views["W6"]["candidate"]])
    late_baseline = np.concatenate([views["W5"]["baseline"], views["W6"]["baseline"]])
    late_actual = y[late_rows]
    late_candidate_mae = float(np.mean(np.abs(late_actual - late_candidate)))
    late_baseline_mae = float(np.mean(np.abs(late_actual - late_baseline)))
    late_gain = late_baseline_mae - late_candidate_mae
    if (len(np.unique(late_rows)) != EXPECTED_LATE_ROWS
            or not np.isclose(late_candidate_mae, float(declared["late_W5_W6"]["candidate_mae"]), atol=1e-12, rtol=0)
            or not np.isclose(late_baseline_mae, float(declared["late_W5_W6"]["champion_mae"]), atol=1e-12, rtol=0)
            or not np.isclose(late_gain, float(declared["late_W5_W6"]["gain_vs_champion"]), atol=1e-12, rtol=0)):
        raise ExpandedCampaignError("Independent W5+W6 regression metric mismatch")
    gates = {
        "primary_gain_at_least_0_03": metric_rows["primary"]["gain_vs_champion"] >= 0.03,
        "H12_harm_at_most_0_05": metric_rows["H12"]["gain_vs_champion"] >= -0.05,
        "at_least_four_of_six_windows_nonnegative": sum(
            metric_rows[f"W{i}"]["gain_vs_champion"] >= 0.0 for i in range(1, 7)
        ) >= 4,
        "W5_W6_gain_positive": late_gain > 0.0,
        "all_supported_group_harm_at_most_0_25": not group_failures,
    }
    if gates != declared["qualification"]["gates"]:
        raise ExpandedCampaignError("Independent regression gate arithmetic differs")
    verified["views"] = metric_rows
    verified["groups"] = {"support_minimum": reg_report.MIN_GROUP_SUPPORT,
                           "maximum_allowed_harm": reg_report.MAX_GROUP_HARM}
    verified["late_W5_W6"] = {"n": int(len(late_rows)), "candidate_mae": late_candidate_mae,
                              "champion_mae": late_baseline_mae, "gain_vs_champion": late_gain}
    verified["qualification"] = {"gates": gates, "qualified": all(gates.values()),
                                  "supported_group_failures": [list(x) for x in group_failures]}
    verified["status"] = "PASS"
    verified["test_data_used"] = False
    return verified


def _evaluate_new_candidates() -> dict[str, Any]:
    identity, cls_context, reg_context = _verify_frozen_preflight()
    freeze_signature = json_sha256(identity)
    artifacts = _active_card_artifacts()
    results: dict[str, Any] = {}
    verification: dict[str, Any] = {}
    for card in NEW_CANDIDATES:
        is_classification = card in {"C17", "C11"}
        context = cls_context if is_classification else reg_context
        records = _records_for_candidate(card, cls_context, reg_context, artifacts, freeze_signature)
        origin_receipts = []
        for cutoff, record in records.items():
            metadata = record["metadata"]
            state = metadata.get("state", {})
            origin_receipts.append({
                "cutoff": int(cutoff), "fit_rows": int(metadata.get("fit_rows", state.get("fit_rows", 0))),
                "score_rows": int(len(record["score_rows"])),
                "fit_predictor_count": int(metadata.get("fit_predictor_count", 0)),
                "bundle_sha256": record.get("bundle_sha256"),
                "sidecar_sha256": record.get("sidecar_sha256"),
                "state_sha256": state.get("state_sha256"),
                "reused_pilot": cutoff in PILOT_CUTOFFS,
                "test_data_used": False,
            })
        if is_classification:
            candidate_result, candidate_verify = _classification_candidate_report(
                card, context["frame"], np.asarray(context["target"], dtype=np.int64),
                records, freeze_signature,
            )
            qualification = candidate_result["qualification"]["status"]
        else:
            from scripts import optimization_v3_final_continuation_r22 as reg_report
            cycles = np.asarray(context["cycles"], dtype=np.int64)
            views = _regression_views(context["frame"], cycles, records)
            metrics = reg_report._policy_metrics(
                context["frame"], np.asarray(context["target"], dtype=np.float64), views, "candidate",
            )
            qualification = reg_report._qualification(metrics)
            candidate_result = {
                "card_id": card, "candidate": card, "task": "regression",
                "status": "CANDIDATE_METRICS_READY_FOR_INDEPENDENT_QA",
                "freeze_signature": freeze_signature, "views": metrics["views"],
                "late_W5_W6": metrics["late_W5_W6"], "diagnostics": metrics["diagnostics"],
                "groups": metrics["groups"], "qualification": {
                    **qualification,
                    "status": "QUALIFIED" if qualification["qualified"] else "EXPERIMENTAL_GATE_EXCEPTION",
                },
                "metric_formula": "mean(abs(threat_rank - prediction)); gain = champion MAE - candidate MAE",
                "outcome_reuse_warning": "All evaluation labels are previously used temporal development data; not an untouched holdout.",
                "test_data_used": False, "sample_submission_read": False,
            }
            candidate_verify = _verify_regression_metrics_independently(
                context["frame"], np.asarray(context["target"], dtype=np.float64),
                views, records, candidate_result,
            )
        candidate_result["fit_origin_receipts"] = origin_receipts
        candidate_result["continuation_fit_accounting"] = {
            "new_fit_cutoffs": list(NEW_CUTOFFS), "reused_pilot_cutoffs": list(PILOT_CUTOFFS),
            "new_predictor_fits": 36 if card == "R11" else 4,
            "test_data_used": False,
        }
        candidate_verify["candidate"] = card
        results[card] = candidate_result
        verification[card] = candidate_verify
        candidate_dir = REPORT_DIR / card
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_or_match(candidate_dir / "results.json", candidate_result)
        _write_or_match(candidate_dir / "verification.json", candidate_verify)

    campaign = {
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "freeze_signature": freeze_signature,
        "status": "NEW_CANDIDATE_EVALUATION_COMPLETE",
        "new_candidates": list(NEW_CANDIDATES),
        "candidates": {card: {
            "qualification": results[card]["qualification"]["status"],
            "primary": results[card]["views"]["primary"],
            "H12": results[card]["views"]["H12"],
            "late_W5_W6": results[card].get("late_W5_W6", {}),
            "qualification_gates": results[card]["qualification"]["gates"],
            "verification_status": verification[card].get("status", "PASS"),
            "verification_sha256": json_sha256(verification[card]),
            "result_sha256": json_sha256(results[card]),
        } for card in NEW_CANDIDATES},
        "evaluation_scope": {"pooled_primary_unique_rows": EXPECTED_PRIMARY_ROWS,
                             "primary_cycles": [56, 79], "windows": _json_clean(WINDOWS),
                             "H12_rows": EXPECTED_H12_ROWS,
                             "late_W5_W6_unique_rows": EXPECTED_LATE_ROWS,
                             "labels_previously_used_in_development": True},
        "test_data_used": False, "public_score_used": False,
    }
    _write_or_match(CAMPAIGN_REPORT_PATH, campaign)
    summary_path = REPORT_DIR / "CAMPAIGN_SUMMARY.md"
    lines = ["# V3 expanded continuation results", "",
             f"Freeze signature: `{freeze_signature}`", "",
             "All candidate evaluations use the six predeclared disjoint windows and previously used Train development labels. The H12 diagnostic is built from pilot origins 55/67; it is not independent of primary.", "",
             "| Candidate | Gate status | Primary | H12 | W5+W6 gain | Windows nonnegative | Verification |",
             "|---|---|---:|---:|---:|---:|---|"]
    for card in NEW_CANDIDATES:
        row = campaign["candidates"][card]
        primary = row["primary"]
        h12 = row["H12"]
        late = row["late_W5_W6"]
        nonnegative = sum(
            results[card]["views"][f"W{i}"]["gain_positive_is_better"] >= 0.0
            if card in {"C17", "C11"} else results[card]["views"][f"W{i}"]["gain_vs_champion"] >= 0.0
            for i in range(1, 7)
        )
        if card in {"C17", "C11"}:
            p_gain = primary["gain_positive_is_better"]
            h_gain = h12["gain_positive_is_better"]
            late_gain = late["gain_positive_is_better"]
        else:
            p_gain = primary["gain_vs_champion"]
            h_gain = h12["gain_vs_champion"]
            late_gain = late["gain_vs_champion"]
        lines.append(f"| {card} | {row['qualification']} | {p_gain:.6f} | {h_gain:.6f} | {late_gain:.6f} | {nonnegative}/6 | {row['verification_status']} |")
    lines += ["", "Performance gate failures remain historical failures and are eligible only for the explicitly authorized experimental benchmark exception. Integrity, recipe, leakage, resource, budget, and competition-rule failures are never waived.", ""]
    summary_text = "\n".join(lines)
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary_text:
            raise ExpandedCampaignError("Existing expanded campaign summary differs; refusing overwrite")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text, encoding="utf-8", newline="\n")
    return campaign


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    clean = _json_clean(value)
    if path.exists():
        if _read_json(path) != clean:
            raise ExpandedCampaignError(f"Existing output differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, clean)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    args = parser.parse_args()
    if args.mode == "prepare":
        result = _build_frozen_preflight(write=True)
    elif args.mode == "run":
        result = run_continuations()
    else:
        result = _evaluate_new_candidates()
    print(json.dumps(_json_clean(result), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
