"""Four-fit learned-factor Classification pilot; query labels open after verification only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

import optimization_v3_positional_bag_offset_v1 as offset_inputs
from src.models.learned_pair_pilot_v1 import (initialize_parameters,
                                               objective_gradient_sum,
                                               score_logits, unpack_parameters)

CARD_ID = "LEARNED_PAIR_PILOT_V1"
STAGE = "LEARNED_PAIR_MECHANISM_PILOT"
AMENDMENT_ID = "LEARNED_PAIR_PILOT_CAP113_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = (
    "LEARNED_PAIR_PILOT_V1_E67_FM",
    "LEARNED_PAIR_PILOT_V1_E67_NOPAIR",
    "LEARNED_PAIR_PILOT_V1_E75_FM",
    "LEARNED_PAIR_PILOT_V1_E75_NOPAIR",
)
ORIGINS = (67, 75)
TRAIN = ROOT / "data/raw/classification/Train.csv"
RECIPE = ROOT / "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json"
AUTH = ROOT / "reports/optimization_v3/learned_pair_preflight_v1/SOL_PILOT_AUTHORIZATION.md"
PREFLIGHT_REPORT = ROOT / "reports/optimization_v3/learned_pair_preflight_v1/REPORT.md"
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/LEARNED_PAIR_PILOT_V1"
REPORT_ROOT = ROOT / "reports/optimization_v3/learned_pair_pilot_v1"
ACCOUNTING = ROOT / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json"
A02_REFERENCE_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
VERIFY_SCRIPT = ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v1.py"
INITIAL_CAPS = {"continuation": 38, "control_replay": 2, "pilot": 68, "technical_retry": 1}
AMENDED_CAPS = {"continuation": 38, "control_replay": 2, "pilot": 72, "technical_retry": 1}
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
RANK = 16
CHUNK_ROWS = 1024
C_VALUE = 1.0
SOLVER_OPTIONS = {"maxiter": 1000, "maxfun": 2000, "maxcor": 10,
                  "maxls": 50, "ftol": 1e-9, "gtol": 1e-4}
SOURCE_FILES = (
    "scripts/optimization_v3_learned_pair_pilot_v1.py",
    "scripts/verify_optimization_v3_learned_pair_pilot_v1.py",
    "src/models/learned_pair_pilot_v1.py",
    "tests/test_learned_pair_pilot_v1.py",
    "reports/optimization_v3/learned_pair_pilot_v1/PROTOCOL.md",
    "reports/optimization_v3/learned_pair_preflight_v1/REPORT.md",
    "reports/optimization_v3/learned_pair_preflight_v1/SOL_PILOT_AUTHORIZATION.md",
    "scripts/optimization_v3_positional_bag_offset_v1.py",
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
)


class LearnedPairPilotError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                                   indent=2, allow_nan=False).encode("utf-8") + b"\n")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _atomic_bytes(path, stream.getvalue())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _create_once_receipt(path: Path, payload: Mapping[str, Any], stable_fields: tuple[str, ...]) -> dict[str, Any]:
    """Write an audit receipt once, or validate its immutable identity on resume."""
    body = dict(payload)
    if path.is_file():
        existing = _read_json(path)
        mismatches = [key for key in stable_fields if existing.get(key) != body.get(key)]
        if mismatches:
            raise LearnedPairPilotError(f"Existing audit receipt identity mismatch for {mismatches}: {path}")
        return existing
    _atomic_json(path, body)
    return body


def _git(*args: str) -> str:
    completed = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                               capture_output=True, text=True)
    return completed.stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {rel: _sha256_file(ROOT / rel) for rel in SOURCE_FILES}


def _recipe_body() -> dict[str, Any]:
    return {
        "schema_version": 1, "card_id": CARD_ID, "fit_ids": list(FIT_IDS),
        "origins": list(ORIGINS), "matrix_arm": "saved C2_B fit_base/score_base; X unchanged",
        "interaction": "FM on row-L2-normalized copy Q of X; rank16",
        "score": "Xw+b+0.5*sum((QV)^2-Q^2(V^2))",
        "loss": "SUM stable binary logistic loss + (||w||2^2+||V||F^2+b^2)/(2*C), C=1",
        "chunk_rows": CHUNK_ROWS, "l2_application": "once after sum data loss/gradient accumulation",
        "factor_initialization": {"seed": 42, "distribution": "Normal(0,0.01)", "reset_per_origin": True},
        "main_initialization": {"w": 0.0, "b": 0.0}, "solver": {"method": "L-BFGS-B", **SOLVER_OPTIONS},
        "control": "same X, objective, C, solver and zero w/b; no V",
        "inference": "direct present-stream expert; exact saved C1_B HGB fallback for missing/blank; threshold=0.50",
        "forbidden": ["offset", "A02 residual", "probability blend", "calibration", "Test/Sample/Kaggle", "Regression labels"],
        "views": {"E67": ["W4", "W5", "W6", "H12_E67"], "E75": ["W6"]},
        "gates": {
            "E67_H12_gain_min_vs_bag": 0.0005,
            "E67_H12_gain_min_vs_nopair": 0.0002,
            "E75_W6_gain_min_vs_both": 0.0,
            "E67_window_harm_max_vs_both": 0.005,
            "supported_subgroup_f1_harm_max": 0.01,
            "supported_subgroup_positive_recall_harm_max": 0.02,
            "subgroup_support": {"n": 200, "each_class": 30},
        },
        "resource": {"jobs": 1, "threads": 2, "rss_limit_bytes": RSS_LIMIT,
                     "initial_free_min_bytes": INITIAL_FREE_MIN, "free_reserve_bytes": FREE_RESERVE},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "scipy": scipy.__version__, "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__, "psutil": psutil.__version__},
        "query_targets": "closed until all four successful receipts, durable predictions, and independent PASS",
        "selection_scope": "adaptive diagnostic mechanism pilot; review only; no champion change, continuation, or upload",
    }


def _resource_snapshot(estimate_bytes: int, *, initial: bool = False) -> dict[str, Any]:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    projected_rss = rss + int(estimate_bytes)
    projected_free = available - int(estimate_bytes)
    snapshot = {
        "jobs": 1, "threads": 2, "process_rss_bytes_sample": rss,
        "system_available_ram_bytes_sample": available,
        "estimated_incremental_memory_bytes": int(estimate_bytes),
        "projected_rss_bytes": projected_rss, "projected_available_ram_bytes": projected_free,
        "max_rss_bytes": RSS_LIMIT, "initial_available_ram_min_bytes": INITIAL_FREE_MIN,
        "minimum_free_ram_bytes": FREE_RESERVE,
        "measurement_note": "psutil instantaneous sample and conservative estimate; not OS peak RSS",
    }
    if initial and available < INITIAL_FREE_MIN:
        raise LearnedPairPilotError(f"Initial free RAM below 4 GiB: {available}")
    if projected_rss > RSS_LIMIT:
        raise LearnedPairPilotError(f"Projected process RSS exceeds 6 GiB: {projected_rss}")
    if projected_free < FREE_RESERVE:
        raise LearnedPairPilotError(f"Projected free RAM below 2 GiB reserve: {projected_free}")
    return snapshot


def _resource_check() -> None:
    state = _resource_snapshot(512 * 1024**2)
    if state["projected_rss_bytes"] > RSS_LIMIT or state["projected_available_ram_bytes"] < FREE_RESERVE:
        raise LearnedPairPilotError("Running resource reserve violated")


def _ledger_events(path: Path = LEDGER) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_ledger(event: Mapping[str, Any], *, path: Path, official: bool = False) -> dict[str, Any]:
    dest = path.resolve()
    if dest == LEDGER.resolve() and not official:
        raise LearnedPairPilotError("Production ledger writer is disabled without official authorization")
    if official and dest != LEDGER.resolve():
        raise LearnedPairPilotError("Official ledger mode cannot write to a temporary ledger")
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = _canonical_bytes(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return payload


def _live_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    return offset_inputs.ledger_snapshot(events)


def _origin_files(origin: int) -> tuple[Path, Path, Path]:
    manifest = PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
    pred = PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    return manifest, pred, PILOT_ROOT / f"E{origin}"


def _verify_saved_inputs() -> dict[str, Any]:
    if not TRAIN.is_file() or _sha256_file(TRAIN) != _read_json(RECIPE)["input"]["sha256"]:
        raise LearnedPairPilotError("Official Classification Train hash mismatch")
    state = offset_inputs.verify_saved_inputs(include_current_code=True)
    frame = offset_inputs._read_identity_frame()
    accounting = _read_json(ACCOUNTING)
    by_id = {row["fit_id"]: row for row in accounting["fit_units"]}
    origins: dict[str, Any] = {}
    for origin in ORIGINS:
        manifest_path, pred_path, _ = _origin_files(origin)
        manifest = _read_json(manifest_path)
        if manifest_path.name not in {f"MATRIX_MANIFEST_E{origin}.json"}:
            raise LearnedPairPilotError("Unexpected matrix manifest identity")
        with np.load(pred_path, allow_pickle=False) as data:
            pred = {key: data[key].copy() for key in data.files}
        fit_rows = np.asarray(pred["fit_rows"], dtype=np.int64)
        score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
        if len(np.intersect1d(fit_rows, score_rows)):
            raise LearnedPairPilotError(f"E{origin} fit and score rows overlap")
        if (not np.array_equal(pred["score_record_ids"].astype(str),
                               frame.iloc[score_rows]["record_id"].astype(str).to_numpy())
                or not np.array_equal(pred["score_cycles"].astype(np.int64),
                                      frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64))):
            raise LearnedPairPilotError(f"E{origin} cached row IDs/cycles no longer align")
        fit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        population = by_id[fit_id]["fit_population"]
        if (population.get("row_indices_sha256") != offset_inputs.sha256_array(fit_rows)
                or population.get("matrix_manifest_sha256") != _sha256_file(manifest_path)):
            raise LearnedPairPilotError(f"E{origin} training row/matrix lineage mismatch")
        matrices = {}
        matrix_meta = {}
        for matrix_name in ("fit_base", "score_base"):
            matrices[matrix_name] = offset_inputs._load_matrix(manifest, matrix_name)
            matrix_meta[matrix_name] = {
                "path": manifest["matrices"][matrix_name]["path"],
                "archive_sha256": manifest["matrices"][matrix_name]["sha256"],
                "logical_sha256": manifest["matrices"][matrix_name]["logical_sha256"],
                "shape": list(matrices[matrix_name].shape),
            }
        fit = sparse.csr_matrix(matrices["fit_base"], dtype=np.float64)
        score = sparse.csr_matrix(matrices["score_base"], dtype=np.float64)
        if (fit.shape[0] != len(fit_rows) or score.shape[0] != len(score_rows)
                or fit.shape[1] != score.shape[1]):
            raise LearnedPairPilotError(f"E{origin} C2_B matrices are not row/column aligned")
        present = np.asarray(pred["present_mask"], dtype=bool)
        fallback = np.asarray(pred["prob_missing_hgb"], dtype=np.float64)
        bag = np.asarray(pred["prob_bag"], dtype=np.float64)
        if (present.shape != (len(score_rows),) or fallback.shape != present.shape or bag.shape != present.shape
                or not np.isfinite(fallback).all() or not np.isfinite(bag).all()
                or not np.array_equal(bag[~present], fallback[~present])):
            raise LearnedPairPilotError(f"E{origin} saved missing fallback or bag route is not exact")
        if ((fallback < 0).any() or (fallback > 1).any() or (bag < 0).any() or (bag > 1).any()):
            raise LearnedPairPilotError(f"E{origin} saved baseline probability out of range")
        a02_path = A02_REFERENCE_ROOT / f"E{origin}" / "observed" / "predictions.npz"
        if a02_path.is_file():
            with np.load(a02_path, allow_pickle=False) as a02_archive:
                a02_rows = np.asarray(a02_archive["score_rows"], dtype=np.int64)
                a02_ids = np.asarray(a02_archive["score_record_ids"]).astype(str)
                a02_cycles = np.asarray(a02_archive["score_cycles"], dtype=np.int64)
                a02_current = np.asarray(a02_archive["baseline_prediction"], dtype=np.float64)
                a02_observed = np.asarray(a02_archive["prediction"], dtype=np.float64)
            if (not np.array_equal(a02_rows, score_rows)
                    or not np.array_equal(a02_ids, pred["score_record_ids"].astype(str))
                    or not np.array_equal(a02_cycles, pred["score_cycles"].astype(np.int64))
                    or a02_current.shape != bag.shape or a02_observed.shape != bag.shape
                    or not np.isfinite(a02_current).all() or not np.isfinite(a02_observed).all()
                    or (a02_current < 0).any() or (a02_current > 1).any()
                    or (a02_observed < 0).any() or (a02_observed > 1).any()):
                raise LearnedPairPilotError(f"E{origin} cached C16/A02 references do not exactly align")
            a02_meta = {
                "status": "ALIGNED_REFERENCE_ONLY",
                "path": a02_path.relative_to(ROOT).as_posix(), "bundle_sha256": _sha256_file(a02_path),
                "current_reference_name": "C16_SPARSE25_CACHED_BASELINE",
                "current_probability_sha256": offset_inputs.sha256_array(a02_current),
                "a02_reference_name": "A02_OBSERVED_CACHED_PREDICTION",
                "a02_probability_sha256": offset_inputs.sha256_array(a02_observed),
                "score_ids_sha256": offset_inputs.sha256_array(a02_ids),
                "score_cycles_sha256": offset_inputs.sha256_array(a02_cycles),
            }
        else:
            a02_meta = {"status": "UNKNOWN_REFERENCE_NOT_CACHED"}
        origins[f"E{origin}"] = {
            "origin": origin, "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "matrix_manifest_sha256": _sha256_file(manifest_path),
            "prediction_bundle_path": pred_path.relative_to(ROOT).as_posix(),
            "prediction_bundle_sha256": _sha256_file(pred_path),
            "fit_rows": len(fit_rows), "score_rows": len(score_rows),
            "fit_rows_sha256": offset_inputs.sha256_array(fit_rows),
            "score_rows_sha256": offset_inputs.sha256_array(score_rows),
            "fit_ids_sha256": offset_inputs.sha256_array(frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()),
            "score_ids_sha256": offset_inputs.sha256_array(pred["score_record_ids"].astype(str)),
            "score_cycles_sha256": offset_inputs.sha256_array(pred["score_cycles"].astype(np.int64)),
            "fit_matrix": matrix_meta["fit_base"], "score_matrix": matrix_meta["score_base"],
            "fallback_probability_sha256": offset_inputs.sha256_array(fallback),
            "baseline_probability_sha256": offset_inputs.sha256_array(bag),
            "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
            "text_load_q75_fit_only": manifest.get("text_load_q75_fit_only"),
            "a02_current_references": a02_meta,
        }
    return {
        "input_sha256": _sha256_file(TRAIN), "input_rows": len(frame),
        "historical_freeze_signature": state["pilot_freeze_signature"],
        "historical_recipe_sha256": state["pilot_recipe_sha256"],
        "historical_accounting_sha256": state["pilot_accounting_sha256"],
        "historical_source_hashes": state["source_hashes"], "origins": origins,
    }


def _expected_pre_amend_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "global_cap": 109, "effective_category_caps": dict(INITIAL_CAPS),
        "active_reservations_by_category": dict(INITIAL_CAPS), "active_reservations": 109,
        "fit_starts": 109, "fit_saved": 106, "fit_failed": 3, "fit_pending": 0,
        "event_rows": snapshot["event_rows"],
    }


def preflight() -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise LearnedPairPilotError("Official preflight requires a clean committed worktree")
    if not AUTH.is_file() or not PREFLIGHT_REPORT.is_file():
        raise LearnedPairPilotError("Authorization/preflight report missing")
    synthetic_path = RUN_ROOT / "SYNTHETIC_QA.json"
    if not synthetic_path.is_file():
        raise LearnedPairPilotError("Run the temporary end-to-end synthetic lifecycle on the committed code first")
    synthetic = _read_json(synthetic_path)
    if (synthetic.get("status") != "PASS_SYNTHETIC_END_TO_END"
            or synthetic.get("official_fit_calls") != 0
            or synthetic.get("source_commit") != _git("rev-parse", "HEAD")
            or synthetic.get("source_hashes") != _source_hashes()
            or synthetic.get("live_budget_ledger_sha256_before") != _sha256_file(LEDGER)
            or synthetic.get("live_budget_ledger_sha256_after") != _sha256_file(LEDGER)):
        raise LearnedPairPilotError("Synthetic QA is stale, incomplete, or changed the live ledger")
    state = _verify_saved_inputs()
    events = _ledger_events()
    snapshot = _live_snapshot(events)
    expected = _expected_pre_amend_snapshot(snapshot)
    if snapshot != expected:
        raise LearnedPairPilotError(f"Live ledger checkpoint differs from authorized 109-fit state: {snapshot}")
    if any(e.get("fit_id") in FIT_IDS for e in events if e.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"}):
        raise LearnedPairPilotError("Learned-pair fit IDs already appear in live ledger")
    memory = _resource_snapshot(1024**3, initial=True)
    result = {
        "card_id": CARD_ID, "status": "PASS_READY_TO_FREEZE",
        "source_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "source_hashes": _source_hashes(), "saved_inputs": state,
        "ledger_sha256": _sha256_file(LEDGER), "ledger_snapshot": snapshot,
        "resource": memory, "fit_ids": list(FIT_IDS), "authorization_sha256": _sha256_file(AUTH),
        "synthetic_qa_sha256": _sha256_file(synthetic_path),
    }
    _atomic_json(RUN_ROOT / "PREFLIGHT.json", result)
    return result


def freeze() -> dict[str, Any]:
    existing = RUN_ROOT / "FREEZE.json"
    if existing.is_file():
        return load_freeze()
    result = preflight()
    body = {
        "schema_version": 1, "card_id": CARD_ID,
        "source_commit": result["source_commit"], "branch": result["branch"],
        "input_sha256": result["saved_inputs"]["input_sha256"],
        "source_hashes": result["source_hashes"], "saved_inputs": result["saved_inputs"],
        "recipe": _recipe_body(), "recipe_sha256": _canonical_sha256(_recipe_body()),
        "authorization_sha256": _sha256_file(AUTH),
        "synthetic_qa_sha256": result["synthetic_qa_sha256"],
        "pre_reservation_ledger_sha256": result["ledger_sha256"],
        "pre_reservation_ledger_snapshot": result["ledger_snapshot"],
        "preflight_resource": result["resource"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["freeze_signature"] = _canonical_sha256(body)
    _atomic_json(existing, body)
    return body


def load_freeze() -> dict[str, Any]:
    path = RUN_ROOT / "FREEZE.json"
    if not path.is_file():
        raise LearnedPairPilotError("Frozen protocol missing; run freeze after clean commit")
    frozen = _read_json(path)
    body = dict(frozen)
    signature = body.pop("freeze_signature", None)
    if signature != _canonical_sha256(body):
        raise LearnedPairPilotError("Freeze signature mismatch")
    if (_git("rev-parse", "HEAD") != frozen["source_commit"]
            or _git("branch", "--show-current") != frozen["branch"]
            or _git("status", "--porcelain")):
        raise LearnedPairPilotError("Source commit, branch, or clean worktree changed after freeze")
    if _source_hashes() != frozen["source_hashes"] or _sha256_file(TRAIN) != frozen["input_sha256"]:
        raise LearnedPairPilotError("Frozen source or Train hash changed")
    if _sha256_file(RUN_ROOT / "SYNTHETIC_QA.json") != frozen["synthetic_qa_sha256"]:
        raise LearnedPairPilotError("Synthetic lifecycle QA receipt changed after freeze")
    current = _verify_saved_inputs()
    if current != frozen["saved_inputs"]:
        raise LearnedPairPilotError("Frozen saved input/matrix/fallback lineage changed")
    return frozen


def _own_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {name: [e for e in events if e.get("event") == name and e.get("fit_id") in FIT_IDS]
            for name in ("FIT_STARTED", "FIT_SAVED", "FIT_FAILED")}


def _reserve_locked() -> dict[str, Any]:
    frozen = load_freeze()
    events = _ledger_events()
    snap = _live_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) > 1 or len(reservations) > 1:
        raise LearnedPairPilotError("Duplicate learned-pair amendment/reservation")
    if not amendments and not reservations:
        if (snap != frozen["pre_reservation_ledger_snapshot"]
                or _sha256_file(LEDGER) != frozen["pre_reservation_ledger_sha256"]):
            raise LearnedPairPilotError("Live ledger changed after freeze; refusing new reservation")
        amendment = {
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sam via Sol explicit learned_pair pilot authorization",
            "prior_global_cap": 109, "global_cap": 113,
            "prior_effective_caps": dict(INITIAL_CAPS), "effective_caps": dict(AMENDED_CAPS),
            "active_reservations_before": dict(INITIAL_CAPS), "active_total_before": 109,
            "historical_started_saved_failed_before": {"started": 109, "saved": 106, "failed": 3},
            "pre_amendment_event_rows": len(events),
            "pre_amendment_ledger_sha256": _sha256_file(LEDGER),
            "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
            "reason": "Exactly four authorized E67/E75 FM/no-pair predictor fits; no preprocessing fits or retries.",
        }
        _append_ledger(amendment, path=LEDGER, official=True)
        events = _ledger_events()
        snap = _live_snapshot(events)
    if amendments:
        amendment = amendments[0]
    else:
        amendment = next(e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID)
    if any(amendment.get(k) != v for k, v in {
        "prior_global_cap": 109, "global_cap": 113,
        "prior_effective_caps": INITIAL_CAPS, "effective_caps": AMENDED_CAPS,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
    }.items()):
        raise LearnedPairPilotError("Existing cap amendment differs from exact authorization")
    reservation_body = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": 113, "effective_category_cap": 72,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
    }
    if not reservations:
        expected_pre_reserve = {
            "global_cap": 113, "effective_category_caps": dict(AMENDED_CAPS),
            "active_reservations_by_category": dict(INITIAL_CAPS), "active_reservations": 109,
            "fit_starts": 109, "fit_saved": 106, "fit_failed": 3, "fit_pending": 0,
            "event_rows": snap["event_rows"],
        }
        if snap != expected_pre_reserve:
            raise LearnedPairPilotError(f"Post-amendment budget is not at 109 active reservations: {snap}")
        _append_ledger(reservation_body, path=LEDGER, official=True)
        events = _ledger_events()
        snap = _live_snapshot(events)
    else:
        reservation_body = reservations[0]
    if any(reservation_body.get(k) != v for k, v in {
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": 113, "effective_category_cap": 72,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
    }.items()):
        raise LearnedPairPilotError("Existing four-fit reservation differs from exact authorization")
    own = _own_events(events)
    released = sum(int(e.get("unused_fits", 0)) for e in events
                   if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == RESERVATION_KEY)
    expected = {
        "global_cap": 113, "effective_category_caps": dict(AMENDED_CAPS),
        "active_reservations_by_category": {**INITIAL_CAPS, "pilot": 72 - released},
        "active_reservations": 113 - released,
        "fit_starts": 109 + len(own["FIT_STARTED"]),
        "fit_saved": 106 + len(own["FIT_SAVED"]),
        "fit_failed": 3 + len(own["FIT_FAILED"]),
        "fit_pending": len(own["FIT_STARTED"]) - len(own["FIT_SAVED"]) - len(own["FIT_FAILED"]),
        "event_rows": snap["event_rows"],
    }
    if snap != expected:
        raise LearnedPairPilotError(f"Post-reservation budget accounting differs: {snap} != {expected}")
    record = {"reservation": reservation_body, "snapshot": snap,
              "ledger_sha256": _sha256_file(LEDGER), "freeze_signature": frozen["freeze_signature"]}
    _atomic_json(RUN_ROOT / "RESERVATION.json", record)
    return record


def reserve() -> dict[str, Any]:
    with offset_inputs._fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        return _reserve_locked()


def _check_live(*, freeze_signature: str, allow_pending: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = _ledger_events()
    snapshot = _live_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT" and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED" and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) != 1 or len(reservations) != 1:
        raise LearnedPairPilotError("Budget amendment/reservation must each occur exactly once")
    amendment, reservation = amendments[0], reservations[0]
    if (amendment.get("freeze_signature") != freeze_signature or reservation.get("freeze_signature") != freeze_signature
            or reservation.get("fit_ids") != list(FIT_IDS)):
        raise LearnedPairPilotError("Live ledger reservation does not match the active freeze")
    own = _own_events(events)
    terminals = {e.get("fit_id") for e in own["FIT_SAVED"] + own["FIT_FAILED"]}
    pending = [e for e in own["FIT_STARTED"] if e.get("fit_id") not in terminals]
    released = sum(int(e.get("unused_fits", 0)) for e in events
                   if e.get("event") == "RESERVATION_UNUSED_RELEASED" and e.get("reservation_key") == RESERVATION_KEY)
    expected = {
        "global_cap": 113, "effective_category_caps": dict(AMENDED_CAPS),
        "active_reservations_by_category": {**INITIAL_CAPS, "pilot": 72 - released},
        "active_reservations": 113 - released,
        "fit_starts": 109 + len(own["FIT_STARTED"]),
        "fit_saved": 106 + len(own["FIT_SAVED"]),
        "fit_failed": 3 + len(own["FIT_FAILED"]),
        "fit_pending": len(pending), "event_rows": snapshot["event_rows"],
    }
    if snapshot != expected:
        raise LearnedPairPilotError(f"Shared live budget differs from exact pilot accounting: {snapshot}")
    allowed_pending_count = 1 if allow_pending is not None else 0
    if len(pending) > allowed_pending_count or (allow_pending not in (None, "*")
                                                  and any(e.get("fit_id") != allow_pending for e in pending)):
        raise LearnedPairPilotError("Unexpected pending fit in shared budget ledger")
    return events, snapshot


@contextmanager
def _rss_sampler(interval: float = 0.5):
    stop = threading.Event()
    process = psutil.Process()
    samples = {"sample_count": 0, "peak_rss_bytes_sample": int(process.memory_info().rss),
               "minimum_available_ram_bytes_sample": int(psutil.virtual_memory().available)}

    def sample() -> None:
        while not stop.wait(interval):
            try:
                samples["sample_count"] += 1
                samples["peak_rss_bytes_sample"] = max(samples["peak_rss_bytes_sample"],
                                                         int(process.memory_info().rss))
                samples["minimum_available_ram_bytes_sample"] = min(
                    samples["minimum_available_ram_bytes_sample"], int(psutil.virtual_memory().available))
            except psutil.Error:
                return

    thread = threading.Thread(target=sample, name="learned-pair-rss-sampler", daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join(timeout=max(1.0, 2 * interval))
        samples["peak_rss_bytes_sample"] = max(samples["peak_rss_bytes_sample"], int(process.memory_info().rss))
        samples["minimum_available_ram_bytes_sample"] = min(
            samples["minimum_available_ram_bytes_sample"], int(psutil.virtual_memory().available))
        samples["measurement_note"] = "0.5-second process/system samples; not an OS peak RSS counter"


def _load_origin_for_fit(frozen: Mapping[str, Any], origin: int) -> dict[str, Any]:
    current = _verify_saved_inputs()
    if current != frozen["saved_inputs"]:
        raise LearnedPairPilotError("Saved matrix/fallback inputs changed after freeze")
    frame = offset_inputs._read_identity_frame()
    manifest_path, pred_path, _ = _origin_files(origin)
    manifest = _read_json(manifest_path)
    with np.load(pred_path, allow_pickle=False) as archive:
        pred = {key: archive[key].copy() for key in archive.files}
    fit_rows = np.asarray(pred["fit_rows"], dtype=np.int64)
    score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
    fit_ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
    fit_cycles = frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64)
    fit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
    accounting = _read_json(ACCOUNTING)
    by_id = {item["fit_id"]: item for item in accounting["fit_units"]}
    population = by_id[fit_id]["fit_population"]
    y = offset_inputs._fit_labels(TRAIN, fit_rows, fit_ids, population["fit_target_sha256"])
    if np.unique(y).tolist() != [0, 1]:
        raise LearnedPairPilotError(f"E{origin} training population must contain class order [0, 1]")
    x_fit = sparse.csr_matrix(offset_inputs._load_matrix(manifest, "fit_base"), dtype=np.float64)
    x_score = sparse.csr_matrix(offset_inputs._load_matrix(manifest, "score_base"), dtype=np.float64)
    if x_fit.shape[0] != len(y) or x_score.shape[0] != len(score_rows) or x_fit.shape[1] != x_score.shape[1]:
        raise LearnedPairPilotError(f"E{origin} saved matrices and row populations do not align")
    return {
        "origin": origin, "manifest": manifest, "manifest_path": manifest_path,
        "prediction_path": pred_path, "prediction": pred, "frame": frame,
        "fit_rows": fit_rows, "score_rows": score_rows, "fit_ids": fit_ids,
        "fit_cycles": fit_cycles, "x_fit": x_fit, "x_score": x_score, "y": y,
        "fit_target_sha256": population["fit_target_sha256"],
        "fit_population": population,
        "matrix_manifest_sha256": _sha256_file(manifest_path),
    }


def _fit_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def _validate_receipt(fit_id: str, frozen: Mapping[str, Any], *, artifact_path: Path,
                      receipt_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise LearnedPairPilotError(f"Durable fit artifact or receipt missing: {fit_id}")
    receipt = _read_json(receipt_path)
    if (receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != frozen["freeze_signature"]
            or receipt.get("artifact_sha256") != _sha256_file(artifact_path)
            or receipt.get("source_commit") != frozen["source_commit"]):
        raise LearnedPairPilotError(f"Fit receipt lineage/hash mismatch: {fit_id}")
    return receipt


def _resume_started_fit(fit_id: str, frozen: Mapping[str, Any], events: list[dict[str, Any]], *,
                        ledger_path: Path = LEDGER, run_root: Path = RUN_ROOT,
                        official: bool = True, artifact_path: Path | None = None,
                        receipt_path: Path | None = None) -> str:
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) != 1 or len(terminals) > 1:
        raise LearnedPairPilotError(f"Inconsistent fit lifecycle for {fit_id}")
    if terminals:
        if terminals[0]["event"] == "FIT_FAILED":
            return "failed_consumed_no_retry"
        artifact = artifact_path or (run_root / "units" / f"{fit_id}.npz")
        receipt = receipt_path or (run_root / "receipts" / f"{fit_id}.json")
        _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt)
        return "already_saved"
    artifact = artifact_path or (run_root / "units" / f"{fit_id}.npz")
    receipt_path = receipt_path or (run_root / "receipts" / f"{fit_id}.json")
    try:
        receipt = _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
    except Exception as exc:
        failure_path = run_root / "failures" / f"{fit_id}.json"
        failure_record = {
            "fit_id": fit_id, "freeze_signature": frozen["freeze_signature"],
            "source_commit": frozen["source_commit"], "error_type": type(exc).__name__,
            "error": str(exc), "interrupted_before_valid_durable_receipt": True,
            "started_event": starts[0], "artifact_present": artifact.is_file(),
            "receipt_present": receipt_path.is_file(),
        }
        _atomic_json(failure_path, failure_record)
        terminal_receipt = {
            "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id,
            "fit_kind": "PREDICTOR", "optimizer_converged": False,
            "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
            "source_hashes": frozen["source_hashes"], "input_sha256": frozen["input_sha256"],
            "fit_population": starts[0].get("fit_population", {}), "recipe": frozen["recipe"],
            "failure": failure_record,
            "artifact_sha256": _sha256_file(artifact) if artifact.is_file() else None,
            "failure_record_sha256": _sha256_file(failure_path),
        }
        _atomic_json(receipt_path, terminal_receipt)
        _append_ledger({
            "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "fit_kind": "PREDICTOR", "amendment_id": AMENDMENT_ID,
            "failure_receipt": receipt_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": _sha256_file(receipt_path),
            "error": f"interrupted_or_corrupt_no_retry: {type(exc).__name__}: {exc}",
        }, path=ledger_path, official=official)
        return "failed_interrupted_no_retry"
    if not receipt.get("optimizer_converged", False):
        event = "FIT_FAILED"
        status = "failed_optimizer_no_retry"
    else:
        event = "FIT_SAVED"
        status = "resumed_from_durable_receipt"
    _append_ledger({
        "event": event, "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": receipt["origin"], "fit_kind": "PREDICTOR",
        "amendment_id": AMENDMENT_ID, "receipt_sha256": _sha256_file(receipt_path),
        "resumed_from_durable_receipt": True,
    }, path=ledger_path, official=official)
    return status


def _release_unstarted(frozen: Mapping[str, Any], why: str) -> dict[str, Any] | None:
    events = _ledger_events()
    own = _own_events(events)
    started_ids = {e["fit_id"] for e in own["FIT_STARTED"]}
    unstarted = _unstarted_fit_ids(FIT_IDS, started_ids)
    prior = [e for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"
             and e.get("reservation_key") == RESERVATION_KEY]
    if prior:
        expected = sum(int(e.get("unused_fits", 0)) for e in prior)
        if expected != len(unstarted):
            raise LearnedPairPilotError("Prior unused-fit release does not match unstarted IDs")
        return prior[-1]
    if not unstarted:
        return None
    _, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    if snapshot["fit_pending"]:
        raise LearnedPairPilotError("Cannot release fits while a fit is pending")
    event = {
        "event": "RESERVATION_UNUSED_RELEASED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "amendment_id": AMENDMENT_ID, "unused_fits": len(unstarted),
        "fit_ids": unstarted, "freeze_signature": frozen["freeze_signature"],
        "reason": why,
    }
    return _append_ledger(event, path=LEDGER, official=True)


def _unstarted_fit_ids(fit_ids: tuple[str, ...], started_ids: set[str]) -> list[str]:
    return [fit_id for fit_id in fit_ids if fit_id not in started_ids]


def _write_failure_checkpoint(frozen: Mapping[str, Any], reason: str) -> dict[str, Any]:
    completions = [event for event in _ledger_events()
                   if event.get("event") == "CARD_COMPLETED" and event.get("card_id") == CARD_ID]
    if len(completions) > 1:
        raise LearnedPairPilotError("Duplicate terminal failure/completion ledger rows")
    if completions:
        result_path = RUN_ROOT / "RESULTS.json"
        if not result_path.is_file() or _sha256_file(result_path) != completions[0].get("result_sha256"):
            raise LearnedPairPilotError("Existing failure checkpoint does not match CARD_COMPLETED receipt")
        return _read_json(result_path)
    _release_unstarted(frozen, reason)
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "FAIL_OPTIMIZER_INCONCLUSIVE" if own["FIT_FAILED"] else "FAIL_INTERRUPTION_INCONCLUSIVE",
        "query_labels_opened": False, "predictive_gates_evaluated": False,
        "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["input_sha256"], "fit_ids": list(FIT_IDS),
        "fit_saved_ids": [e["fit_id"] for e in own["FIT_SAVED"]],
        "fit_failed_ids": [e["fit_id"] for e in own["FIT_FAILED"]],
        "ledger_snapshot": snapshot, "ledger_sha256": _sha256_file(LEDGER),
        "interpretation": "At least one official predictor did not converge or was interrupted. Its consumed ID was not retried; remaining unstarted IDs were released. No outer/query target was opened and this is inconclusive about the interaction mechanism.",
        "reason": reason,
    }
    result_path = RUN_ROOT / "RESULTS.json"
    _atomic_json(result_path, result)
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    _atomic_bytes(REPORT_ROOT / "RESULTS.md", (
        "# Learned pair pilot — optimizer checkpoint\n\n"
        f"**Status:** `{result['status']}`. No query target opened; no predictive gate evaluated.\n\n"
        f"{result['interpretation']}\n\nReason: `{reason}`.\n"
    ).encode("utf-8"))
    _append_ledger({
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": STAGE, "freeze_signature": frozen["freeze_signature"],
        "status": result["status"], "fit_ids": list(FIT_IDS),
        "result_sha256": _sha256_file(result_path),
    }, path=LEDGER, official=True)
    return result


def _write_fit_terminal(fit_id: str, origin: int, frozen: Mapping[str, Any],
                        data: Mapping[str, Any], theta: np.ndarray, optimizer: Mapping[str, Any],
                        *, success: bool, initial_loss: float, final_loss: float,
                        final_gradient: np.ndarray, started_at: float,
                        resource_before: Mapping[str, Any], resource_after: Mapping[str, Any],
                        rss_samples: Mapping[str, Any]) -> dict[str, Any]:
    candidate = fit_id.endswith("_FM")
    fit_logits, pair_scores = score_logits(theta, data["x_fit"], rank=RANK,
                                           with_factors=candidate, chunk_rows=CHUNK_ROWS)
    w, b, v = unpack_parameters(theta, data["x_fit"].shape[1], RANK, with_factors=candidate)
    artifact_path, receipt_path = _fit_paths(fit_id)
    arrays = {
        "parameters": np.asarray(theta, dtype=np.float64),
        "fit_logits": np.asarray(fit_logits, dtype=np.float64),
        "fit_row_ids": np.asarray(data["fit_ids"]).astype(str),
        "fit_cycles": np.asarray(data["fit_cycles"], dtype=np.int64),
        "class_order": np.asarray([0, 1], dtype=np.int8),
    }
    if pair_scores is not None:
        arrays["fit_pair_scores"] = np.asarray(pair_scores, dtype=np.float64)
    _atomic_npz(artifact_path, **arrays)
    init_theta = initialize_parameters(data["x_fit"].shape[1], RANK, seed=42,
                                       with_factors=candidate)
    init_v = init_theta[data["x_fit"].shape[1] + 1:] if candidate else np.empty(0)
    final_v = theta[data["x_fit"].shape[1] + 1:] if candidate else np.empty(0)
    opt = dict(optimizer)
    opt.update({
        "success": bool(success), "initial_loss_sum": float(initial_loss),
        "final_loss_sum": float(final_loss),
        "final_gradient_inf_norm_sum": float(np.max(np.abs(final_gradient))),
        "final_gradient_inf_norm_mean": float(np.max(np.abs(final_gradient)) / len(data["y"])),
        "factor_initial_norm": float(np.linalg.norm(init_v)),
        "factor_final_norm": float(np.linalg.norm(final_v)),
        "factor_change_norm": float(np.linalg.norm(final_v - init_v)),
        "pair_score_fit_mean": float(np.mean(pair_scores)) if pair_scores is not None else 0.0,
        "pair_score_fit_std": float(np.std(pair_scores)) if pair_scores is not None else 0.0,
        "pair_score_fit_max_abs": float(np.max(np.abs(pair_scores))) if pair_scores is not None else 0.0,
        "parameter_w_norm": float(np.linalg.norm(w)), "intercept": float(b),
    })
    receipt = {
        "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id,
        "origin": origin, "component": "FM_RANK16" if candidate else "LOGISTIC_MAIN_ONLY",
        "fit_kind": "PREDICTOR", "optimizer_converged": bool(success),
        "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
        "source_hashes": frozen["source_hashes"], "input_sha256": frozen["input_sha256"],
        "fit_population": {
            "rows": len(data["y"]), "record_ids_sha256": offset_inputs.sha256_array(data["fit_ids"]),
            "row_indices_sha256": offset_inputs.sha256_array(data["fit_rows"]),
            "fit_target_sha256": data["fit_target_sha256"],
            "matrix_manifest_sha256": data["matrix_manifest_sha256"],
            "target_read_purpose": "fit-prefix-only", "class_order": [0, 1],
        },
        "matrix": {
            "fit_base_archive_sha256": frozen["saved_inputs"]["origins"][f"E{origin}"]["fit_matrix"]["archive_sha256"],
            "fit_base_logical_sha256": frozen["saved_inputs"]["origins"][f"E{origin}"]["fit_matrix"]["logical_sha256"],
            "fit_base_shape": list(data["x_fit"].shape),
            "score_base_archive_sha256": frozen["saved_inputs"]["origins"][f"E{origin}"]["score_matrix"]["archive_sha256"],
        },
        "recipe": frozen["recipe"], "optimizer": opt,
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "raw_fit_logits_sha256": offset_inputs.sha256_array(fit_logits),
        "raw_fit_pair_scores_sha256": offset_inputs.sha256_array(pair_scores) if pair_scores is not None else None,
        "resource_before": dict(resource_before), "resource_after": dict(resource_after),
        "resource_during_fit_sample": dict(rss_samples),
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    _atomic_json(receipt_path, receipt)
    receipt_sha = _sha256_file(receipt_path)
    return {"artifact_path": artifact_path, "receipt_path": receipt_path,
            "artifact_sha256": _sha256_file(artifact_path), "receipt_sha256": receipt_sha,
            "receipt": receipt}


def _fit_one(fit_id: str, frozen: Mapping[str, Any]) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise LearnedPairPilotError(f"Unknown fit ID: {fit_id}")
    origin = 67 if "E67" in fit_id else 75
    data = _load_origin_for_fit(frozen, origin)
    if (not np.isfinite(data["x_fit"].data).all()
            or not np.isfinite(data["x_score"].data).all()):
        raise LearnedPairPilotError(f"E{origin} matrix has non-finite values")
    resource_before = _resource_snapshot(1024**3, initial=False)
    candidate = fit_id.endswith("_FM")
    theta0 = initialize_parameters(data["x_fit"].shape[1], RANK, seed=42,
                                   factor_std=0.01, with_factors=candidate)
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    terminals = {e["fit_id"] for e in own["FIT_SAVED"] + own["FIT_FAILED"]}
    if snapshot["fit_pending"] or len(own["FIT_STARTED"]) >= 4 or fit_id in {e["fit_id"] for e in own["FIT_STARTED"]}:
        raise LearnedPairPilotError("Ledger is not ready for this exact next fit")
    expected_next = next((item for item in FIT_IDS if item not in {e["fit_id"] for e in own["FIT_STARTED"]}), None)
    if expected_next != fit_id:
        raise LearnedPairPilotError(f"Next authorized fit is {expected_next}, not {fit_id}")
    fit_record = {
        "rows": len(data["y"]), "record_ids_sha256": offset_inputs.sha256_array(data["fit_ids"]),
        "row_indices_sha256": offset_inputs.sha256_array(data["fit_rows"]),
        "fit_target_sha256": data["fit_target_sha256"],
        "matrix_manifest_sha256": data["matrix_manifest_sha256"],
        "origin": origin, "component": "FM_RANK16" if candidate else "LOGISTIC_MAIN_ONLY",
        "target_read_purpose": "fit-prefix-only",
    }
    _append_ledger({
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": fit_record["component"],
        "fit_kind": "PREDICTOR", "fit_consumed_at_start": True,
        "amendment_id": AMENDMENT_ID, "freeze_signature": frozen["freeze_signature"],
        "source_commit": frozen["source_commit"], "resource_snapshot": resource_before,
        "fit_population": fit_record,
    }, path=LEDGER, official=True)
    started = time.perf_counter()
    result: Any = None
    initial_loss = float("nan")
    final_loss = float("nan")
    final_grad = np.full_like(theta0, np.nan)
    opt_details: dict[str, Any] = {"method": "L-BFGS-B", **SOLVER_OPTIONS,
                                   "class_order": [0, 1], "initialization_seed": 42,
                                   "factor_std": 0.01, "chunk_rows": CHUNK_ROWS}
    try:
        initial_loss = float("nan")
        with _rss_sampler() as rss_samples, threadpool_limits(limits=2):
            initial_loss, _ = objective_gradient_sum(
                theta0, data["x_fit"], data["y"], rank=RANK,
                with_factors=candidate, chunk_rows=CHUNK_ROWS, resource_check=_resource_check)
            result = minimize(
                lambda parameters: objective_gradient_sum(
                    parameters, data["x_fit"], data["y"], rank=RANK,
                    with_factors=candidate, chunk_rows=CHUNK_ROWS,
                    resource_check=_resource_check),
                theta0, method="L-BFGS-B", jac=True, options=SOLVER_OPTIONS,
            )
            final_loss, final_grad = objective_gradient_sum(
                result.x, data["x_fit"], data["y"], rank=RANK,
                with_factors=candidate, chunk_rows=CHUNK_ROWS, resource_check=_resource_check)
        opt_details.update({
            "status_code": int(result.status), "message": str(result.message),
            "nit": int(result.nit), "nfev": int(result.nfev),
            "njev": int(getattr(result, "njev", result.nfev)),
            "solver_fun_sum": float(result.fun), "solver_gradient_inf_norm_sum": float(np.max(np.abs(result.jac))),
            "solver_gradient_inf_norm_mean": float(np.max(np.abs(result.jac)) / len(data["y"])),
            "verified_final_loss_sum": float(final_loss),
            "verified_final_gradient_inf_norm_sum": float(np.max(np.abs(final_grad))),
        })
        finite_state = (np.isfinite(result.x).all() and np.isfinite(final_loss)
                        and np.isfinite(final_grad).all() and np.isfinite(np.asarray(result.fun)).all())
        converged = bool(result.success and finite_state and final_loss <= initial_loss + 1e-8)
        opt_details["success"] = bool(result.success)
        opt_details["finite_state"] = bool(finite_state)
        opt_details["loss_nonincrease_guard"] = bool(final_loss <= initial_loss + 1e-8)
        resource_after = _resource_snapshot(0)
        with threadpool_limits(limits=2):
            durable = _write_fit_terminal(
                fit_id, origin, frozen, data, np.asarray(result.x, dtype=np.float64), opt_details,
                success=converged, initial_loss=initial_loss, final_loss=final_loss,
                final_gradient=final_grad, started_at=started, resource_before=resource_before,
                resource_after=resource_after, rss_samples=rss_samples,
            )
        event_name = "FIT_SAVED" if converged else "FIT_FAILED"
        _append_ledger({
            "event": event_name, "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
            "amendment_id": AMENDMENT_ID, "receipt_sha256": durable["receipt_sha256"],
            "failure": None if converged else "optimizer_not_converged_or_guard_failed",
        }, path=LEDGER, official=True)
        outcome = {"fit_id": fit_id, "status": "saved" if converged else "failed_optimizer_no_retry",
                   "receipt_sha256": durable["receipt_sha256"], "optimizer": durable["receipt"]["optimizer"]}
        if not converged:
            _write_failure_checkpoint(frozen, f"{fit_id}: {result.message}; nfev={result.nfev}; success={result.success}")
        return outcome
    except Exception as exc:
        # The ID is consumed once FIT_STARTED exists; persist a terminal diagnostic and never retry it.
        artifact_path, receipt_path = _fit_paths(fit_id)
        failure_path = RUN_ROOT / "failures" / f"{fit_id}.json"
        failure_record = {
            "fit_id": fit_id, "origin": origin, "freeze_signature": frozen["freeze_signature"],
            "error_type": type(exc).__name__, "error": str(exc), "source_commit": frozen["source_commit"],
            "artifact_present": artifact_path.is_file(), "receipt_present": receipt_path.is_file(),
            "optimizer": opt_details,
        }
        _atomic_json(failure_path, failure_record)
        failure_receipt = {
            "schema_version": 1, "card_id": CARD_ID, "fit_id": fit_id, "origin": origin,
            "fit_kind": "PREDICTOR", "optimizer_converged": False,
            "freeze_signature": frozen["freeze_signature"], "source_commit": frozen["source_commit"],
            "source_hashes": frozen["source_hashes"], "input_sha256": frozen["input_sha256"],
            "fit_population": fit_record, "recipe": frozen["recipe"],
            "optimizer": opt_details, "failure": failure_record,
            "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha256_file(artifact_path) if artifact_path.is_file() else None,
            "failure_record_sha256": _sha256_file(failure_path),
        }
        _atomic_json(receipt_path, failure_receipt)
        failure_receipt_sha = _sha256_file(receipt_path)
        current_events = _ledger_events()
        terminal = [e for e in current_events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
        if not terminal:
            _append_ledger({
                "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": origin, "fit_kind": "PREDICTOR",
                "amendment_id": AMENDMENT_ID, "failure_receipt": receipt_path.relative_to(ROOT).as_posix(),
                "receipt_sha256": failure_receipt_sha,
                "error": f"{type(exc).__name__}: {exc}",
            }, path=LEDGER, official=True)
        elif terminal[0]["event"] == "FIT_SAVED":
            return {"fit_id": fit_id, "status": "saved", "receipt_sha256": terminal[0].get("receipt_sha256"),
                    "note": "FIT_SAVED was already durable; no additional terminal event appended"}
        _write_failure_checkpoint(frozen, f"{fit_id}: {type(exc).__name__}: {exc}")
        return {"fit_id": fit_id, "status": "failed_consumed_no_retry", "error": str(exc)}


def fit_next() -> dict[str, Any]:
    lock_path = RUN_ROOT / "FIT_RUN.lock"
    with offset_inputs._fit_run_lock(lock_path):
        frozen = load_freeze()
        events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"], allow_pending="*")
        own = _own_events(events)
        pending = [e for e in own["FIT_STARTED"]
                   if e["fit_id"] not in {x["fit_id"] for x in own["FIT_SAVED"] + own["FIT_FAILED"]}]
        if len(pending) > 1:
            raise LearnedPairPilotError("More than one fit is pending; stop for reconciliation")
        if pending:
            resumed = _resume_started_fit(pending[0]["fit_id"], frozen, events)
            if resumed.startswith("failed"):
                return _write_failure_checkpoint(frozen, resumed)
            return {"fit_id": pending[0]["fit_id"], "status": resumed, "optimizer_rerun": False}
        if own["FIT_FAILED"]:
            return _write_failure_checkpoint(frozen, "prior fit has terminal failure; retries and later fits prohibited")
        started_ids = {event["fit_id"] for event in own["FIT_STARTED"]}
        fit_id = next((item for item in FIT_IDS if item not in started_ids), None)
        if fit_id is None:
            return {"status": "all_four_fits_terminal", "snapshot": snapshot}
        return _fit_one(fit_id, frozen)


def _load_origin_unlabeled(frozen: Mapping[str, Any], origin: int) -> dict[str, Any]:
    current = _verify_saved_inputs()
    if current != frozen["saved_inputs"]:
        raise LearnedPairPilotError("Saved matrix/prediction sources changed after freeze")
    frame = offset_inputs._read_identity_frame()
    manifest_path, pred_path, _ = _origin_files(origin)
    manifest = _read_json(manifest_path)
    with np.load(pred_path, allow_pickle=False) as archive:
        pred = {key: archive[key].copy() for key in archive.files}
    score_rows = np.asarray(pred["score_rows"], dtype=np.int64)
    score_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy()
    score_cycles = frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
    if (not np.array_equal(score_ids, pred["score_record_ids"].astype(str))
            or not np.array_equal(score_cycles, pred["score_cycles"].astype(np.int64))):
        raise LearnedPairPilotError(f"E{origin} score identity mismatch")
    matrix = sparse.csr_matrix(offset_inputs._load_matrix(manifest, "score_base"), dtype=np.float64)
    if matrix.shape[0] != len(score_rows):
        raise LearnedPairPilotError(f"E{origin} score matrix row count mismatch")
    return {"manifest": manifest, "manifest_path": manifest_path, "prediction": pred,
            "score_rows": score_rows, "score_ids": score_ids, "score_cycles": score_cycles,
            "x_score": matrix}


def build_predictions() -> dict[str, Any]:
    frozen = load_freeze()
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    if (len(own["FIT_STARTED"]) != 4 or len(own["FIT_SAVED"]) != 4
            or own["FIT_FAILED"] or snapshot["fit_pending"]):
        raise LearnedPairPilotError("All four learned-pair and no-pair predictors must converge before scoring")
    outputs: dict[str, Any] = {}
    for origin in ORIGINS:
        data = _load_origin_unlabeled(frozen, origin)
        pred = data["prediction"]
        present = np.asarray(pred["present_mask"], dtype=bool)
        fallback = np.asarray(pred["prob_missing_hgb"], dtype=np.float64)
        bag = np.asarray(pred["prob_bag"], dtype=np.float64)
        if not np.array_equal(bag[~present], fallback[~present]):
            raise LearnedPairPilotError(f"E{origin} cached missing fallback changed")
        probabilities: dict[str, np.ndarray] = {"bag": bag.copy()}
        a02_ref_meta = frozen["saved_inputs"]["origins"][f"E{origin}"]["a02_current_references"]
        if a02_ref_meta["status"] == "ALIGNED_REFERENCE_ONLY":
            a02_path = ROOT / a02_ref_meta["path"]
            if _sha256_file(a02_path) != a02_ref_meta["bundle_sha256"]:
                raise LearnedPairPilotError(f"E{origin} A02/C16 cached reference changed after freeze")
            with np.load(a02_path, allow_pickle=False) as a02_archive:
                probabilities["current_ref"] = np.asarray(a02_archive["baseline_prediction"], dtype=np.float64)
                probabilities["a02_ref"] = np.asarray(a02_archive["prediction"], dtype=np.float64)
        raw_scores: dict[str, np.ndarray] = {}
        pair_scores: dict[str, np.ndarray] = {}
        fit_receipts: dict[str, Any] = {}
        for suffix in ("FM", "NOPAIR"):
            fit_id = f"{CARD_ID}_E{origin}_{suffix}"
            artifact, receipt_path = _fit_paths(fit_id)
            receipt = _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
            if not receipt.get("optimizer_converged"):
                raise LearnedPairPilotError(f"{fit_id} did not converge")
            with np.load(artifact, allow_pickle=False) as saved:
                theta = np.asarray(saved["parameters"], dtype=np.float64).copy()
            with_factors = suffix == "FM"
            with threadpool_limits(limits=2):
                raw, pair = score_logits(theta, data["x_score"], rank=RANK,
                                         with_factors=with_factors, chunk_rows=CHUNK_ROWS)
            probability_direct = expit(raw)
            if probability_direct.shape != present.shape or not np.isfinite(probability_direct).all():
                raise LearnedPairPilotError(f"{fit_id} produced invalid score probabilities")
            routed = probability_direct.copy()
            routed[~present] = fallback[~present]
            if not np.array_equal(routed[~present], fallback[~present]):
                raise LearnedPairPilotError(f"{fit_id} missing fallback is not bit-exact")
            key = "fm" if with_factors else "nopair"
            probabilities[key] = routed
            raw_scores[key] = raw
            pair_scores[key] = pair if pair is not None else np.zeros_like(raw)
            fit_receipts[fit_id] = {
                "receipt_sha256": _sha256_file(receipt_path),
                "artifact_sha256": _sha256_file(artifact),
                "optimizer": receipt["optimizer"],
            }
        payload = {
            "origin": np.asarray([origin], dtype=np.int64),
            "score_rows": data["score_rows"], "score_record_ids": data["score_ids"],
            "score_cycles": data["score_cycles"], "present_mask": present,
            "prob_bag": probabilities["bag"], "prob_fm": probabilities["fm"],
            "prob_nopair": probabilities["nopair"],
            "prob_missing_hgb": fallback,
            "raw_score_fm": raw_scores["fm"], "raw_score_nopair": raw_scores["nopair"],
            "pair_score_fm": pair_scores["fm"],
        }
        if "current_ref" in probabilities:
            payload["prob_current_c16_reference"] = probabilities["current_ref"]
            payload["prob_a02_observed_reference"] = probabilities["a02_ref"]
        dest = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        _atomic_npz(dest, **payload)
        outputs[f"E{origin}"] = {
            "path": dest.relative_to(ROOT).as_posix(), "sha256": _sha256_file(dest),
            "score_rows": len(data["score_rows"]), "fit_receipts": fit_receipts,
        }
    ready = {
        "schema_version": 1, "card_id": CARD_ID,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["input_sha256"], "query_labels_opened": False,
        "predictions": outputs, "fit_ids": list(FIT_IDS),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(RUN_ROOT / "PREDICTIONS_READY.json", ready)
    return ready


def verify_prelabel() -> dict[str, Any]:
    frozen = load_freeze()
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    if not ready_path.is_file():
        raise LearnedPairPilotError("Durable prediction readiness manifest missing")
    ready = _read_json(ready_path)
    if (ready.get("freeze_signature") != frozen["freeze_signature"]
            or ready.get("query_labels_opened") is not False
            or set(ready.get("predictions", {})) != {"E67", "E75"}):
        raise LearnedPairPilotError("Prediction readiness manifest lineage mismatch")
    completed, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(completed)
    if len(own["FIT_SAVED"]) != 4 or own["FIT_FAILED"] or snapshot["fit_pending"]:
        raise LearnedPairPilotError("All four successful fit receipts must be ledger-complete before independent verification")
    command = [sys.executable, str(VERIFY_SCRIPT), "--official", "--run-root", str(RUN_ROOT)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LearnedPairPilotError(f"Independent prelabel verifier failed: {proc.stderr or proc.stdout}")
    verified = json.loads(proc.stdout)
    if verified.get("status") != "PASS":
        raise LearnedPairPilotError("Independent prelabel verifier did not return PASS")
    verified["freeze_signature"] = frozen["freeze_signature"]
    verified["predictions_ready_sha256"] = _sha256_file(ready_path)
    verified["prediction_output_sha256s"] = {
        key: _sha256_file(ROOT / value["path"]) for key, value in ready["predictions"].items()
    }
    verified["budget_ledger_sha256"] = _sha256_file(LEDGER)
    verified["verification_signature"] = _canonical_sha256(verified)
    _atomic_json(RUN_ROOT / "PRELABEL_VERIFICATION.json", verified)
    return verified


def _assert_prelabel(verified: Mapping[str, Any], frozen: Mapping[str, Any]) -> dict[str, Any]:
    ready_path = RUN_ROOT / "PREDICTIONS_READY.json"
    ready = _read_json(ready_path)
    outputs = {key: _sha256_file(ROOT / value["path"]) for key, value in ready["predictions"].items()}
    body = dict(verified)
    sig = body.pop("verification_signature", None)
    if (verified.get("status") != "PASS"
            or verified.get("freeze_signature") != frozen["freeze_signature"]
            or verified.get("predictions_ready_sha256") != _sha256_file(ready_path)
            or verified.get("prediction_output_sha256s") != outputs
            or sig != _canonical_sha256(body)):
        raise LearnedPairPilotError("Query labels remain closed: prelabel proof is missing/stale")
    return ready


def _label(value: Any) -> str:
    return "<missing>" if pd.isna(value) else str(value)


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise LearnedPairPilotError("Invalid metric probability vector")
    if not len(y):
        return {"n": 0, "positive": 0, "negative": 0, "macro_f1": None,
                "positive_f1": None, "precision": None, "positive_recall": None,
                "roc_auc": None, "average_precision": None, "log_loss": None,
                "brier": None, "confusion_matrix": [[0, 0], [0, 0]]}
    predicted = (p >= 0.5).astype(np.int8)
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int((1 - y).sum()),
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=[0, 1], zero_division=0)),
        "positive_f1": float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "positive_recall": float(recall_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        "average_precision": float(average_precision_score(y, p)) if np.unique(y).size == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def _view_masks(origin: int, cycles: np.ndarray) -> dict[str, np.ndarray]:
    if origin == 67:
        return {"W4": (cycles >= 68) & (cycles <= 71),
                "W5": (cycles >= 72) & (cycles <= 75),
                "W6": (cycles >= 76) & (cycles <= 79),
                "H12_E67": (cycles >= 68) & (cycles <= 79)}
    return {"W6": (cycles >= 76) & (cycles <= 79)}


def _supported(metric: Mapping[str, Any]) -> bool:
    return metric["n"] >= 200 and metric["positive"] >= 30 and metric["negative"] >= 30


def _meets_gate(value: float, threshold: float) -> bool:
    return bool(value >= threshold - 1e-12)


def _gates_for_origin(origin: int, views: list[dict[str, Any]],
                      groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_view = {(item["view"], item["arm"]): item for item in views}
    gates: list[dict[str, Any]] = []

    def delta(view: str, arm: str, control: str, metric: str) -> float:
        a = by_view[(view, arm)][metric]
        b = by_view[(view, control)][metric]
        if a is None or b is None:
            raise LearnedPairPilotError(f"Missing {metric} for {view}/{arm}/{control}")
        return float(a - b)

    if origin == 67:
        for control, minimum in (("bag", 0.0005), ("nopair", 0.0002)):
            value = delta("H12_E67", "fm", control, "macro_f1")
            gates.append({"gate": f"E67_H12_gain_min_{minimum}_vs_{control}",
                          "value": value, "threshold": minimum, "pass": _meets_gate(value, minimum)})
        for view in ("W4", "W5", "W6"):
            for control in ("bag", "nopair"):
                value = delta(view, "fm", control, "macro_f1")
                gates.append({"gate": f"E67_{view}_harm_max_0.005_vs_{control}",
                              "value": value, "threshold": -0.005, "pass": _meets_gate(value, -0.005)})
    else:
        for control in ("bag", "nopair"):
            value = delta("W6", "fm", control, "macro_f1")
            gates.append({"gate": f"E75_W6_gain_nonnegative_vs_{control}",
                          "value": value, "threshold": 0.0, "pass": _meets_gate(value, 0.0)})
    for item in groups:
        if item["arm"] != "fm":
            continue
        support = _supported(item)
        gates.append({"gate": f"{item['view']}_{item['group']}_{item['group_value']}_support",
                      "value": {"n": item["n"], "positive": item["positive"], "negative": item["negative"]},
                      "threshold": {"n": 200, "each_class": 30}, "pass": True if support else None,
                      "status": "SUPPORTED" if support else "UNKNOWN_UNSUPPORTED"})
        if not support:
            continue
        control_item = {(r["view"], r["group"], r["group_value"], r["arm"]): r for r in groups}
        for control in ("bag", "nopair"):
            reference = control_item[(item["view"], item["group"], item["group_value"], control)]
            for key, maximum in (("macro_f1", 0.01), ("positive_recall", 0.02)):
                value = float(item[key] - reference[key])
                gates.append({"gate": f"{item['view']}_{item['group']}_{item['group_value']}_{key}_harm_vs_{control}",
                              "value": value, "threshold": -maximum, "pass": _meets_gate(value, -maximum)})
    return gates


def score() -> dict[str, Any]:
    with offset_inputs._fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        return _score_locked()


def _score_locked() -> dict[str, Any]:
    prior = [e for e in _ledger_events() if e.get("event") == "CARD_COMPLETED" and e.get("card_id") == CARD_ID]
    if len(prior) > 1:
        raise LearnedPairPilotError("Duplicate card completion events")
    if prior:
        path = RUN_ROOT / "RESULTS.json"
        if not path.is_file() or _sha256_file(path) != prior[0].get("result_sha256"):
            raise LearnedPairPilotError("Completed result receipt is missing or ambiguous")
        return _read_json(path)
    frozen = load_freeze()
    verified = _read_json(RUN_ROOT / "PRELABEL_VERIFICATION.json")
    ready = _assert_prelabel(verified, frozen)
    events, snapshot = _check_live(freeze_signature=frozen["freeze_signature"])
    own = _own_events(events)
    if (snapshot["fit_starts"] != 113 or snapshot["fit_saved"] != 110
            or snapshot["fit_failed"] != 3 or snapshot["fit_pending"] != 0
            or len(own["FIT_SAVED"]) != 4 or own["FIT_FAILED"]):
        raise LearnedPairPilotError("All four successful predictors must be saved before opening outer labels")
    scored_identity: dict[str, Any] = {}
    for key, value in ready["predictions"].items():
        with np.load(ROOT / value["path"], allow_pickle=False) as saved_prediction:
            scored_identity[key] = {
                "score_rows_sha256": offset_inputs.sha256_array(saved_prediction["score_rows"]),
                "score_ids_sha256": offset_inputs.sha256_array(saved_prediction["score_record_ids"].astype(str)),
                "score_cycles_sha256": offset_inputs.sha256_array(saved_prediction["score_cycles"].astype(np.int64)),
            }
    access_payload = {
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "evaluation only after durable predictions and independent prelabel PASS",
        "prediction_ready_sha256": _sha256_file(RUN_ROOT / "PREDICTIONS_READY.json"),
        "prelabel_verification_sha256": _sha256_file(RUN_ROOT / "PRELABEL_VERIFICATION.json"),
        "train_sha256": frozen["input_sha256"],
        "columns_opened": ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
        "scored_populations": scored_identity,
    }
    access = _create_once_receipt(
        RUN_ROOT / "QUERY_LABEL_ACCESS.json", access_payload,
        ("purpose", "prediction_ready_sha256", "prelabel_verification_sha256",
         "train_sha256", "columns_opened", "scored_populations"),
    )
    # This is the first outer/query target read; all fit inputs and predictions are already durable.
    train = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
                        dtype={"record_id": "string"})
    read_payload = {
        "read_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_label_access_sha256": _sha256_file(RUN_ROOT / "QUERY_LABEL_ACCESS.json"),
        "train_sha256": frozen["input_sha256"],
        "rows_read": int(len(train)),
        "columns_read": ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
    }
    _create_once_receipt(
        RUN_ROOT / "QUERY_LABEL_READ.json", read_payload,
        ("query_label_access_sha256", "train_sha256", "rows_read", "columns_read"),
    )
    origin_results: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for origin in ORIGINS:
        key = f"E{origin}"
        pred_path = ROOT / ready["predictions"][key]["path"]
        with np.load(pred_path, allow_pickle=False) as archive:
            pred = {name: archive[name].copy() for name in archive.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        sub = train.iloc[rows]
        if not np.array_equal(sub["record_id"].astype(str).to_numpy(), pred["score_record_ids"].astype(str)):
            raise LearnedPairPilotError(f"E{origin} score IDs changed before metrics")
        y = pd.to_numeric(sub["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y, (0, 1)).all():
            raise LearnedPairPilotError(f"E{origin} query labels are non-binary")
        cycles = pd.to_numeric(sub["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        arms = {"bag": pred["prob_bag"], "nopair": pred["prob_nopair"], "fm": pred["prob_fm"]}
        if "prob_current_c16_reference" in pred:
            arms["current_c16_reference"] = pred["prob_current_c16_reference"]
            arms["a02_observed_reference"] = pred["prob_a02_observed_reference"]
        views_map = _view_masks(origin, cycles)
        present = np.asarray(pred["present_mask"], dtype=bool)
        text_load = pd.to_numeric(sub["text_load"], errors="coerce").to_numpy(dtype=np.float64)
        manifest = _read_json(PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json")
        q75 = float(manifest["text_load_q75_fit_only"])
        q4 = np.isfinite(text_load) & (text_load >= q75)
        group_masks: list[tuple[str, str, np.ndarray]] = [
            ("availability", "present", present),
            ("availability", "missing_or_blank", ~present),
            ("text_load", "Q4", q4),
            ("text_load", "below_Q4", np.isfinite(text_load) & ~q4),
            ("text_load", "missing", ~np.isfinite(text_load)),
        ]
        ef = sub["entry_friction"].to_numpy(dtype=object)
        for value in sorted({_label(v) for v in ef}):
            group_masks.append(("entry_friction", value,
                                np.asarray([_label(v) == value for v in ef], dtype=bool)))
        view_metrics: list[dict[str, Any]] = []
        subgroup_metrics: list[dict[str, Any]] = []
        per_cycle: list[dict[str, Any]] = []
        for view, vm in views_map.items():
            for arm, probability in arms.items():
                view_metrics.append({"view": view, "arm": arm, **_metric(y[vm], np.asarray(probability)[vm])})
            for group, value, gm in group_masks:
                for arm, probability in arms.items():
                    subgroup_metrics.append({"view": view, "group": group, "group_value": value,
                                             "arm": arm, **_metric(y[vm & gm], np.asarray(probability)[vm & gm])})
        for cycle in sorted(np.unique(cycles).tolist()):
            cm = cycles == cycle
            each = {arm: _metric(y[cm], np.asarray(probability)[cm]) for arm, probability in arms.items()}
            per_cycle.append({"cycle": int(cycle), "n": int(cm.sum()), "metrics": each,
                              "fm_minus_bag_macro_f1": each["fm"]["macro_f1"] - each["bag"]["macro_f1"],
                              "fm_minus_nopair_macro_f1": each["fm"]["macro_f1"] - each["nopair"]["macro_f1"]})
        gates = _gates_for_origin(origin, view_metrics, subgroup_metrics)
        all_gates.extend({"origin": origin, **gate} for gate in gates)
        origin_results[key] = {
            "score_rows": len(rows), "present_rows": int(present.sum()), "missing_or_blank_rows": int((~present).sum()),
            "fit_defined_text_load_q75": q75, "views": view_metrics,
            "subgroups": subgroup_metrics, "per_cycle": per_cycle, "gates": gates,
            "per_cycle_positive_gain_counts": {
                "fm_vs_bag": sum(row["fm_minus_bag_macro_f1"] > 0 for row in per_cycle),
                "fm_vs_nopair": sum(row["fm_minus_nopair_macro_f1"] > 0 for row in per_cycle),
                "cycles": len(per_cycle),
            },
            "a02_current_references": frozen["saved_inputs"]["origins"][key]["a02_current_references"],
        }
    fit_receipts: dict[str, Any] = {}
    optimizer_pass = True
    for fit_id in FIT_IDS:
        artifact, receipt_path = _fit_paths(fit_id)
        receipt = _validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
        fit_receipts[fit_id] = {"receipt_sha256": _sha256_file(receipt_path),
                                "artifact_sha256": _sha256_file(artifact),
                                "optimizer": receipt["optimizer"]}
        optimizer_pass &= bool(receipt["optimizer_converged"])
    unsupported_groups = [gate for gate in all_gates if gate.get("status") == "UNKNOWN_UNSUPPORTED"]
    evaluable_gates = [gate for gate in all_gates if gate.get("pass") is not None]
    all_pass = optimizer_pass and all(gate["pass"] is True for gate in evaluable_gates)
    result = {
        "schema_version": 1, "card_id": CARD_ID,
        "status": "PASS_MECHANISM_REVIEW_ONLY" if all_pass else "FAIL_DIAGNOSTIC_ONLY",
        "adaptive_selection": True, "mechanism_pilot": True,
        "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
        "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["input_sha256"],
        "query_labels_opened_after_prelabel_verification": True,
        "query_label_access_receipt_sha256": _sha256_file(RUN_ROOT / "QUERY_LABEL_ACCESS.json"),
        "query_label_read_receipt_sha256": _sha256_file(RUN_ROOT / "QUERY_LABEL_READ.json"),
        "prelabel_verification_sha256": _sha256_file(RUN_ROOT / "PRELABEL_VERIFICATION.json"),
        "predictions_ready_sha256": _sha256_file(RUN_ROOT / "PREDICTIONS_READY.json"),
        "ledger_snapshot": snapshot, "ledger_sha256": _sha256_file(LEDGER),
        "fit_receipts": fit_receipts, "optimizer_all_success": optimizer_pass,
        "gate_checks": all_gates, "all_gates_pass": all_pass,
        "evaluable_gates_pass": all_pass, "unsupported_group_checks": unsupported_groups,
        "unsupported_group_check_count": len(unsupported_groups),
        "origins": origin_results,
        "interpretation": "Adaptive two-origin learned-factor mechanism pilot. E75 W6 overlaps E67 W6 and is a history-sensitivity view, not independent replication. PASS is for mechanism review only; it does not change the champion, authorize continuation, or authorize upload.",
    }
    result_path = RUN_ROOT / "RESULTS.json"
    _atomic_json(result_path, result)
    report = _make_report(result)
    _atomic_json(REPORT_ROOT / "RESULTS.json", result)
    _atomic_bytes(REPORT_ROOT / "RESULTS.md", report.encode("utf-8"))
    _append_ledger({
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": STAGE, "freeze_signature": frozen["freeze_signature"],
        "status": result["status"], "fit_ids": list(FIT_IDS),
        "result_sha256": _sha256_file(result_path),
    }, path=LEDGER, official=True)
    return result


def _make_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Learned pair pilot — hasil diagnostic", "",
        f"**Status: `{result['status']}`.** Kandidat tetap review mekanisme; tidak mengganti champion dan tidak mengizinkan continuation atau upload.", "",
        "## Rancangan dan batas inferensi", "",
        "Pilot adaptif ini menguji rank-16 factorization machine dengan objective binary logistic SUM + satu penalti L2 pada parameter utama, faktor, dan intercept. Kontrol no-pair memakai objective, inisialisasi w/b, dan solver yang sama. Present stream memakai expert langsung; missing/blank memakai probabilitas HGB tersimpan secara bit-exact. Tidak ada offset, residual A02, blending, atau kalibrasi.",
        "E67 H12 adalah view agregat utama. E67 W4/W5/W6 dan E75 W6 diperlakukan sebagai view temporal/history sensitivity; W6 E75 tumpang tindih dengan W6 E67. Ini bukan bukti holdout baru atau replikasi independen.", "",
        "## Metrik dan gate", "",
        "Macro F1 memakai threshold 0,50. Gate mengikuti otorisasi pilot: E67 H12 +0,0005 vs cached C2_B dan +0,0002 vs matched no-pair; E75 W6 non-negatif vs keduanya; E67 tiap window harm maksimal 0,005; subgroup yang didukung F1 harm maksimal 0,01 dan positive-recall harm maksimal 0,02.", "",
        "| Origin | View | Arm | n | Macro F1 | Pos recall | Precision | ROC AUC | AP | Log loss | Brier | Confusion [[TN,FP],[FN,TP]] |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for origin, block in result["origins"].items():
        for item in block["views"]:
            roc = f"{item['roc_auc']:.6f}" if item["roc_auc"] is not None else "NA"
            ap = f"{item['average_precision']:.6f}" if item["average_precision"] is not None else "NA"
            lines.append(f"| {origin} | {item['view']} | {item['arm']} | {item['n']} | {item['macro_f1']:.6f} | {item['positive_recall']:.6f} | {item['precision']:.6f} | {roc} | {ap} | {item['log_loss']:.6f} | {item['brier']:.6f} | {item['confusion_matrix']} |")
    lines.extend(["", "### Gate results", "", "| Origin | Gate | Value | Threshold | Result |", "|---|---|---:|---:|---|"])
    for gate in result["gate_checks"]:
        value = gate["value"] if isinstance(gate["value"], (float, int)) else json.dumps(gate["value"], sort_keys=True)
        lines.append(f"| {gate['origin']} | {gate['gate']} | {value} | {gate['threshold']} | {'PASS' if gate['pass'] else gate.get('status', 'FAIL')} |")
    lines.extend(["", "### Per-cycle metrics and direction", "", "| Origin | Cycle | Arm | n | Macro F1 | Precision | Pos recall | Confusion [[TN,FP],[FN,TP]] | FM Δ vs bag | FM Δ vs no-pair |", "|---|---:|---|---:|---:|---:|---:|---|---:|---:|"])
    for origin, block in result["origins"].items():
        for row in block["per_cycle"]:
            for arm, metric in row["metrics"].items():
                d_bag = row["fm_minus_bag_macro_f1"] if arm == "fm" else ""
                d_control = row["fm_minus_nopair_macro_f1"] if arm == "fm" else ""
                lines.append(f"| {origin} | {row['cycle']} | {arm} | {row['n']} | {metric['macro_f1']:.6f} | {metric['precision']:.6f} | {metric['positive_recall']:.6f} | {metric['confusion_matrix']} | {d_bag} | {d_control} |")
    lines.extend(["", "Positive/negative cycle deltas are shown directly for FM. C16 Sparse25 and A02 observed rows are descriptive cached references only; the frozen C2_B bag and same-solver no-pair remain the gate comparators.", ""])
    lines.extend(["", "### Supported subgroup guards", "", "Full subgroup metrics and unsupported groups are in `RESULTS.json`. Candidate and controls use the same exact missing-stream fallback.", ""])
    for origin, block in result["origins"].items():
        lines.append(f"#### {origin}")
        for item in block["subgroups"]:
            if item["arm"] == "fm":
                lines.append(f"- {item['view']} / {item['group']}={item['group_value']}: n={item['n']} (positive={item['positive']}, negative={item['negative']}), Macro F1={item['macro_f1']}, recall={item['positive_recall']}; {'supported' if _supported(item) else 'unsupported'}")
    lines.extend(["", "## Solver and resource receipts", "", "Solver `nit/nfev/status`, sum/mean gradient norms, initial/final loss, factor movement, pair-score contribution, exact source hashes, and sampled resource observations are recorded per fit in `RESULTS.json` and fit receipts. Resource maxima are process/system samples, not OS peak RSS. Cached references are row/ID/cycle checked when available; missing references are UNKNOWN. They are never fitted or used by the pilot gates.", ""])
    lines.append("## Kesimpulan")
    lines.append("Interpretasi dibatasi pada pilot adaptif ini. Bahkan bila seluruh gate lolos, hasil hanya layak untuk review mekanisme berikutnya; bukan champion baru, bukan validasi independen, bukan izin refit lanjutan, dan bukan izin submission.")
    return "\n".join(lines) + "\n"


def run_synthetic_lifecycle(temp_root: Path) -> dict[str, Any]:
    """Exercise optimizer, persistence, resume, independent verification, and guards on temporary data only."""
    temp_root = Path(temp_root).resolve()
    if temp_root == ROOT.resolve() or temp_root.is_relative_to(ROOT.resolve()):
        raise LearnedPairPilotError("Synthetic output root must be outside the repository")
    if temp_root.exists() and any(temp_root.iterdir()):
        raise LearnedPairPilotError(f"Synthetic root must be new/empty: {temp_root}")
    temp_root.mkdir(parents=True, exist_ok=True)
    live_hash_before = _sha256_file(LEDGER)
    report_path = temp_root / "SYNTHETIC_REPORT.json"
    fit_call_count = 0
    result: dict[str, Any] = {}
    try:
        rng = np.random.default_rng(321)
        dense = rng.normal(0, 1.0, (36, 6))
        dense[np.abs(dense) < 0.55] = 0.0
        fit_x = sparse.csr_matrix(dense)
        w_truth = np.array([0.18, -0.25, 0.12, 0.0, 0.16, -0.1])
        v_truth = np.random.default_rng(17).normal(0.0, 0.18, (6, 2))
        from src.models.learned_pair_pilot_v1 import fm_pair_score, row_l2_normalize
        latent = np.asarray(fit_x @ w_truth).reshape(-1) + fm_pair_score(row_l2_normalize(fit_x), v_truth)
        y = ((latent + rng.normal(0, 0.9, len(latent))) > np.median(latent)).astype(np.int8)
        if np.unique(y).tolist() != [0, 1]:
            raise LearnedPairPilotError("Synthetic target generation did not produce both classes")
        theta0 = initialize_parameters(6, 2, seed=42, factor_std=0.01)
        initial_loss, _ = objective_gradient_sum(theta0, fit_x, y, rank=2, chunk_rows=7)
        fit_call_count += 1
        with threadpool_limits(limits=2):
            synthetic_result = minimize(
                lambda theta: objective_gradient_sum(theta, fit_x, y, rank=2, chunk_rows=7),
                theta0, method="L-BFGS-B", jac=True,
                options={"maxiter": 500, "maxfun": 2000, "maxcor": 10,
                         "maxls": 50, "ftol": 1e-9, "gtol": 1e-4},
            )
        final_loss, final_grad = objective_gradient_sum(synthetic_result.x, fit_x, y, rank=2, chunk_rows=7)
        if (not synthetic_result.success or not np.isfinite(synthetic_result.x).all()
                or not np.isfinite(final_loss) or not np.isfinite(final_grad).all()
                or final_loss > initial_loss + 1e-8):
            raise LearnedPairPilotError(f"Synthetic L-BFGS-B convergence gate failed: {synthetic_result.message}")
        fit_logits, _ = score_logits(synthetic_result.x, fit_x, rank=2, chunk_rows=7)
        score_dense = np.array([
            [0.0, 1.0, -0.5, 0.0, 1.5, 0.2],
            [1.0, 0.0, 0.0, -1.0, 0.5, 0.0],
            [0.2, -0.8, 1.2, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-0.4, 0.7, 0.0, 0.8, 0.0, -0.2],
            [0.9, 0.0, -0.3, 0.1, 0.0, 1.0],
            [0.0, -0.2, 0.4, 0.6, -0.8, 0.0],
        ], dtype=np.float64)
        score_x = sparse.csr_matrix(score_dense)
        raw, pair = score_logits(synthetic_result.x, score_x, rank=2, chunk_rows=3)
        direct = expit(raw)
        present = np.array([True, False, True, False, True, True, False])
        fallback = np.array([0.31, 0.62, 0.21, 0.49, 0.7, 0.12, 0.88])
        routed = direct.copy()
        routed[~present] = fallback[~present]
        record_ids = np.array(["id-z", "id-a", "id-m", "id-b", "id-y", "id-c", "id-x"])
        cycles = np.array([79, 68, 76, 71, 75, 69, 78], dtype=np.int64)
        score_matrix_path = temp_root / "matrix.npz"
        fit_matrix_path = temp_root / "fit_matrix.npz"
        fit_y_path = temp_root / "fit_y.npy"
        sparse.save_npz(score_matrix_path, score_x)
        sparse.save_npz(fit_matrix_path, fit_x)
        with fit_y_path.open("wb") as stream:
            np.save(stream, y, allow_pickle=False)
        artifact_path = temp_root / "parameters.npz"
        _atomic_npz(artifact_path, parameters=np.asarray(synthetic_result.x, dtype=np.float64),
                    fit_logits=fit_logits, fit_row_ids=np.asarray([f"fit-{i:02d}" for i in range(len(y))]),
                    fit_cycles=np.asarray([(i * 7) % 13 + 47 for i in range(len(y))], dtype=np.int64))
        prediction_path = temp_root / "prediction.npz"
        _atomic_npz(prediction_path, raw_score=raw, probability_direct=direct,
                    probability_routed=routed, present_mask=present,
                    fallback_probability=fallback, record_ids=record_ids,
                    cycles=cycles, class_order=np.array([0, 1], dtype=np.int8))
        with_factors = True
        matrix_logical = offset_inputs.sha256_csr(score_x)
        fit_matrix_logical = offset_inputs.sha256_csr(fit_x)
        optimizer_receipt = {
            "success": bool(synthetic_result.success), "status_code": int(synthetic_result.status),
            "message": str(synthetic_result.message), "nit": int(synthetic_result.nit),
            "nfev": int(synthetic_result.nfev), "verified_final_loss_sum": float(final_loss),
            "verified_final_gradient_inf_norm_sum": float(np.max(np.abs(final_grad))),
        }
        manifest = {
            "matrix_archive_sha256": _sha256_file(score_matrix_path),
            "matrix_logical_sha256": matrix_logical, "matrix_shape": list(score_x.shape),
            "fit_matrix_name": fit_matrix_path.name,
            "fit_matrix_archive_sha256": _sha256_file(fit_matrix_path),
            "fit_matrix_logical_sha256": fit_matrix_logical,
            "fit_y_name": fit_y_path.name, "fit_y_sha256": _sha256_file(fit_y_path),
            "rank": 2, "chunk_rows": 7, "with_factors": True,
            "record_ids": record_ids.tolist(), "cycles": cycles.tolist(),
            "fit_record_ids": [f"fit-{i:02d}" for i in range(len(y))],
            "fit_cycles": [(i * 7) % 13 + 47 for i in range(len(y))],
        }
        manifest_path = temp_root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        receipt = {
            "fit_id": "SYNTHETIC_LEARNED_PAIR", "origin": "synthetic",
            "freeze_signature": "synthetic-freeze", "source_commit": "synthetic-source",
            "optimizer_converged": True, "optimizer": optimizer_receipt,
            "artifact_sha256": _sha256_file(artifact_path),
            "prediction_sha256": _sha256_file(prediction_path),
        }
        receipt_path = temp_root / "receipt.json"
        _atomic_json(receipt_path, receipt)
        ledger_path = temp_root / "ledger.jsonl"
        _append_ledger({"event": "FIT_STARTED", "fit_id": "SYNTHETIC_LEARNED_PAIR"}, path=ledger_path)
        frozen_test = {"freeze_signature": "synthetic-freeze", "source_commit": "synthetic-source"}
        events = _ledger_events(ledger_path)
        first = _resume_started_fit("SYNTHETIC_LEARNED_PAIR", frozen_test, events,
                                    ledger_path=ledger_path, run_root=temp_root, official=False,
                                    artifact_path=artifact_path, receipt_path=receipt_path)
        second = _resume_started_fit("SYNTHETIC_LEARNED_PAIR", frozen_test, _ledger_events(ledger_path),
                                     ledger_path=ledger_path, run_root=temp_root, official=False,
                                     artifact_path=artifact_path, receipt_path=receipt_path)
        if first != "resumed_from_durable_receipt" or second != "already_saved":
            raise LearnedPairPilotError(f"Synthetic durable resume failed: {first}, {second}")
        import verify_optimization_v3_learned_pair_pilot_v1 as verifier
        verified = verifier.verify_synthetic_bundle(temp_root)
        if verified.get("status") != "PASS":
            raise LearnedPairPilotError("Independent synthetic verifier did not PASS")
        tamper_root = temp_root / "tamper_check"
        if tamper_root.exists():
            raise LearnedPairPilotError(f"Tamper test path already exists: {tamper_root}")
        if temp_root not in tamper_root.resolve().parents:
            raise LearnedPairPilotError("Tamper test directory escaped the explicit synthetic root")
        import shutil
        shutil.copytree(temp_root, tamper_root)
        with (tamper_root / "matrix.npz").open("ab") as stream:
            stream.write(b"tampered")
        try:
            verifier.verify_synthetic_bundle(tamper_root)
        except Exception:
            tamper_rejected = True
        else:
            tamper_rejected = False
        shutil.rmtree(tamper_root)
        if not tamper_rejected:
            raise LearnedPairPilotError("Independent verifier accepted a tampered matrix")
        production_writer_blocked = False
        before_guard_hash = _sha256_file(LEDGER)
        try:
            _append_ledger({"event": "SHOULD_NOT_WRITE"}, path=LEDGER, official=False)
        except LearnedPairPilotError:
            production_writer_blocked = True
        if not production_writer_blocked or _sha256_file(LEDGER) != before_guard_hash:
            raise LearnedPairPilotError("Production ledger guard failed or changed live ledger")
        live_hash_after = _sha256_file(LEDGER)
        if live_hash_after != live_hash_before:
            raise LearnedPairPilotError("Live fit ledger changed during temporary synthetic QA")
        temp_events = _ledger_events(ledger_path)
        _append_ledger({"event": "CARD_COMPLETED", "status": "PASS_SYNTHETIC"}, path=ledger_path)
        result = {
            "status": "PASS_SYNTHETIC_END_TO_END", "live_budget_ledger_sha256_before": live_hash_before,
            "live_budget_ledger_sha256_after": live_hash_after,
            "source_commit": _git("rev-parse", "HEAD"), "source_hashes": _source_hashes(),
            "official_fit_calls": 0, "synthetic_optimizer_fit_calls": fit_call_count,
            "synthetic_optimizer": optimizer_receipt,
            "factor_movement_norm": float(np.linalg.norm(synthetic_result.x[7:] - theta0[7:])),
            "durable_artifact_sha256": _sha256_file(artifact_path),
            "durable_prediction_sha256": _sha256_file(prediction_path),
            "resume_first": first, "resume_second": second,
            "optimizer_reruns_on_resume": 0,
            "independent_verification": verified,
            "tampered_matrix_rejected": tamper_rejected,
            "production_writer_blocked": production_writer_blocked,
            "temporary_ledger_events_before_card_complete": [event["event"] for event in temp_events],
            "temporary_ledger_events_final": [event["event"] for event in _ledger_events(ledger_path)],
            "report_path": report_path.as_posix(),
        }
        _atomic_json(report_path, result)
        return result
    finally:
        if _sha256_file(LEDGER) != live_hash_before:
            raise LearnedPairPilotError("Live ledger hash changed during synthetic lifecycle; investigate immediately")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "reserve", "fit-next",
                                            "predictions", "verify-prelabel", "score", "synthetic"))
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            value = preflight()
        elif args.command == "freeze":
            value = freeze()
        elif args.command == "reserve":
            value = reserve()
        elif args.command == "fit-next":
            value = fit_next()
        elif args.command == "predictions":
            value = build_predictions()
        elif args.command == "verify-prelabel":
            value = verify_prelabel()
        elif args.command == "score":
            value = score()
        else:
            if args.temp_root is None:
                parser.error("synthetic requires explicit --temp-root outside the repository")
                return 2
            value = run_synthetic_lifecycle(args.temp_root)
            _atomic_json(RUN_ROOT / "SYNTHETIC_QA.json", value)
        print(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
