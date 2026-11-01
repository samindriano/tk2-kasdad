"""Frozen continuation execution and independent evaluation for R22 only.

This module is a continuation-only path. It reuses the original R22 pilot
artifacts at cutoffs 55/67 and fits only the four predeclared origins 59/63/71/75.
It never resolves or reads an official Test file.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r01 as r01
from scripts import optimization_v3_batch1_r22 as r22_pilot
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    file_sha256,
    json_sha256,
    read_bundle,
    stream_missing_mask,
    write_bundle_exclusive,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r22 import CARD_ID, FEATURE_NAMES
from src.models.optimization_v3_final_continuation_r22 import (
    CONTINUATION_CUTOFFS,
    fit_raw_predict,
    postprocess_policies,
)


EXP_DIR = ROOT / "experiments/optimization_v3/final_continuation/R22"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_final_continuation/R22"
REPORT_DIR = ROOT / "reports/optimization_v3/final_continuation/R22"
RECIPE_PATH = EXP_DIR / "recipe.json"
FREEZE_PATH = EXP_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
RESULT_PATH = REPORT_DIR / "result.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUMMARY_PATH = REPORT_DIR / "R22_CONTINUATION.md"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_batch1/batch_ledger.jsonl"
TRAINING_LOCK = ROOT / "experiments/runs/optimization_v3_batch1/.training.lock"

DECISION_PATH = ROOT / "reports/optimization_v3/final_continuation_plan/DECISION.md"
PROMPT_PATH = ROOT / "reports/optimization_v3/final_continuation_plan/LUNA_PROMPT.md"
GUARD_AUDIT_PATH = ROOT / "reports/optimization_v3/final_continuation_plan/guard_audit.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R22.md"
FULLSIX_CACHE = ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
FULLSIX_REPORT = ROOT / "reports/optimization_v2/regression_repr/fullsix.json"
PILOT_FREEZE = ROOT / "experiments/optimization_v3/batch1/R22/freeze.json"
PILOT_BUNDLE_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R22"
PILOT_VERIFIED = {
    "55": {
        "npz": "7978b1326d251586d226daeafaa6412626028964bdeedc4a9688e61f1db930d7",
        "json": "ad60bac203748ea3d50c4b6f6b33b77e94de0a0507e1e293cd0ca4a364e49716",
    },
    "67": {
        "npz": "cf8e02933e4fc9138b378fa1352ee64625484c8795cbdecc34b9d9604bd1d722",
        "json": "32ad552d372fe6f8024b65fadfd66354bf188565c64c8e4126923e1ce14537dd",
    },
}

TRAIN_SHA256 = r22_pilot.TRAIN_SHA256
FULLSIX_CACHE_SHA256 = "4b87b0da34069ea3d75490233c574fbd92283f8b6472a43d8c654a20a4b904c9"
EXPECTED_PRIMARY_ROWS = 38_588
EXPECTED_LATE_ROWS = 15_953
MIN_GROUP_SUPPORT = 100
MAX_GROUP_HARM = 0.25
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
MAX_SUCCESSFUL_FITS = 180
JOINT_RESERVATION_KEY = "FINAL_CONTINUATION:C16_R22"
JOINT_RESERVATION_CARD = "C16+R22"

WINDOWS: dict[str, tuple[int, int, int]] = {
    "W1": (55, 56, 59),
    "W2": (59, 60, 63),
    "W3": (63, 64, 67),
    "W4": (67, 68, 71),
    "W5": (71, 72, 75),
    "W6": (75, 76, 79),
}
NEW_WINDOWS = {name: values for name, values in WINDOWS.items() if values[0] in CONTINUATION_CUTOFFS}
REQUIRED_VIEWS = ("primary", "H12", "W1", "W2", "W3", "W4", "W5", "W6")

SOURCE_PATHS = [
    Path(__file__),
    ROOT / "src/models/optimization_v3_final_continuation_r22.py",
    ROOT / "tests/test_optimization_v3_final_continuation_r22.py",
    ROOT / "src/models/optimization_v3_batch1_r22.py",
    ROOT / "scripts/optimization_v3_batch1_r22.py",
    ROOT / "src/models/optimization_v3_batch1.py",
    ROOT / "scripts/optimization_v3_batch1.py",
    ROOT / "scripts/optimization_v3_batch1_verify.py",
    ROOT / "scripts/optimization_v3_batch1_r01.py",
    ROOT / "src/models/optimization_v3_batch1_r01.py",
    ROOT / "src/models/regression_research_text.py",
    ROOT / "src/models/classification_models.py",
    ROOT / "src/utils/paths.py",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(a: Any, b: Any) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def _resource_preflight(fit_rows: int, available_bytes: int | None = None) -> dict[str, Any]:
    # Same conservative memory policy as the R22 pilot: five dense float64
    # design copies, a histogram allowance, and 2 GiB fixed overhead.
    width = 807 + 32 + 1
    design_bytes = int(fit_rows) * width * 8
    hist_bytes = width * 256 * 16 * THREADS * 3
    fixed_bytes = 2 * 1024**3
    estimate = 5 * design_bytes + hist_bytes + fixed_bytes
    if available_bytes is None:
        available_bytes = int(psutil.virtual_memory().available)
    allowed = min(JOB_CAP_BYTES, max(0, int(available_bytes) - FREE_RAM_RESERVE_BYTES))
    return {
        "status": "PASS" if estimate <= allowed else "BLOCKED_RESOURCE",
        "fit_rows": int(fit_rows),
        "feature_width_upper_bound": width,
        "estimated_peak_bytes": estimate,
        "estimated_peak_is_observed": False,
        "histogram_buffer_allowance_bytes": hist_bytes,
        "fixed_overhead_bytes": fixed_bytes,
        "available_physical_bytes": int(available_bytes),
        "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "job_cap_bytes": JOB_CAP_BYTES,
        "allowed_bytes": allowed,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
    }


def _resource_identity(resources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Strip live RAM/status fields; the freeze binds static policy only."""
    excluded = {"status", "available_physical_bytes", "allowed_bytes"}
    return {str(cutoff): {k: v for k, v in item.items() if k not in excluded}
            for cutoff, item in resources.items()}


def _take_by_rows(rows_source: np.ndarray, values: np.ndarray, rows_wanted: np.ndarray,
                  *, label: str) -> np.ndarray:
    source = np.asarray(rows_source, dtype=np.int64)
    data = np.asarray(values)
    wanted = np.asarray(rows_wanted, dtype=np.int64)
    if source.ndim != 1 or data.shape[:1] != source.shape or len(np.unique(source)) != len(source):
        raise BatchIntegrityError(f"R22 {label} source rows are invalid")
    lookup = {int(row): i for i, row in enumerate(source)}
    try:
        positions = np.asarray([lookup[int(row)] for row in wanted], dtype=np.int64)
    except KeyError as exc:
        raise BatchIntegrityError(f"R22 {label} cache omits source row {exc.args[0]}") from exc
    return data[positions]


def _control_sources() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                                 dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = r01._read_npz(r01.BASE_CACHE)
    sparse_pilot = r01._read_npz(r01.SPARSE_CACHE)
    fullsix = r01._read_npz(FULLSIX_CACHE)
    diagnostic = r01._read_npz(r01.DIAGNOSTIC_CACHE)
    return base, sparse_pilot, fullsix, diagnostic


def _rank50_for_controls(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.shape != cycles.shape or raw.ndim != 1 or not np.isfinite(raw).all():
        raise BatchIntegrityError("R22 cached sparse raw control is invalid")
    ranked = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        pct = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        ranked[mask] = np.clip(0.5 * raw[mask] + 0.5 * pct, 0.0, 100.0)
    return ranked


def _build_primary_controls(frame: pd.DataFrame, cycles: np.ndarray,
                            diagnostic: dict[str, np.ndarray],
                            base: dict[str, np.ndarray],
                            sparse_pilot: dict[str, np.ndarray],
                            fullsix: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    diag_rows = np.asarray(diagnostic["primary_rows"], dtype=np.int64)
    diag_lookup = {int(row): i for i, row in enumerate(diag_rows)}
    if len(diag_lookup) != len(diag_rows):
        raise BatchIntegrityError("R22 primary diagnostic rows contain duplicates")
    output: dict[int, dict[str, np.ndarray]] = {}
    all_rows: list[int] = []
    for window, (cutoff, low, high) in WINDOWS.items():
        rows = np.flatnonzero((cycles >= low) & (cycles <= high)).astype(np.int64)
        expected_count = len(rows)
        if expected_count == 0:
            raise BatchIntegrityError(f"R22 {window} has no Train score rows")
        base_rows = np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)
        if cutoff in (55, 67):
            expected_base_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        else:
            expected_base_rows = rows
        if not np.array_equal(base_rows, expected_base_rows):
            raise BatchIntegrityError(f"R22 paired-control cache row coverage differs at origin {cutoff}")

        if cutoff in (55, 67):
            sparse_rows = np.asarray(sparse_pilot[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
            sparse_raw_source = np.asarray(sparse_pilot[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
            expected_sparse_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        else:
            sparse_rows = np.asarray(fullsix[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
            sparse_raw_source = np.asarray(fullsix[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
            expected_sparse_rows = rows
        if not np.array_equal(sparse_rows, expected_sparse_rows):
            raise BatchIntegrityError(f"R22 sparse-control cache row coverage differs at origin {cutoff}")

        lgb_all = np.asarray(base[f"lgb_{cutoff}"], dtype=np.float64)
        ridge_all = np.asarray(base[f"ridge_{cutoff}"], dtype=np.float64)
        lgb = _take_by_rows(base_rows, lgb_all, rows, label=f"LGB origin {cutoff}")
        ridge = _take_by_rows(base_rows, ridge_all, rows, label=f"Ridge origin {cutoff}")
        sparse_raw = _take_by_rows(sparse_rows, sparse_raw_source, rows, label=f"sparse raw origin {cutoff}")
        score_cycles = cycles[rows]
        sparse_rank50 = _rank50_for_controls(sparse_raw, score_cycles)
        champion = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank50, 0.0, 100.0)

        try:
            diagnostic_positions = np.asarray([diag_lookup[int(row)] for row in rows], dtype=np.int64)
        except KeyError as exc:
            raise BatchIntegrityError(f"R22 diagnostic champion omits primary row {exc.args[0]}") from exc
        checks = {
            "rows": np.array_equal(diag_rows[diagnostic_positions], rows),
            "lgb_svd32": np.array_equal(lgb, np.asarray(diagnostic["primary_lgb_svd32"])[diagnostic_positions]),
            "hybrid_ridge": np.array_equal(ridge, np.asarray(diagnostic["primary_hybrid_ridge"])[diagnostic_positions]),
            "sparse_rank50": np.array_equal(sparse_rank50, np.asarray(diagnostic["primary_sparse_lgb_rank50"])[diagnostic_positions]),
            "champion": np.array_equal(champion, np.asarray(diagnostic["primary_prediction"])[diagnostic_positions]),
        }
        if not all(checks.values()):
            raise BatchIntegrityError(f"R22 cached champion reconstruction failed at {window}: {checks}")
        output[cutoff] = {
            "rows": rows,
            "record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            "cycle_codes": score_cycles,
            "baseline_prediction": champion,
            "baseline_lgb_svd32": lgb,
            "baseline_hybrid_ridge": ridge,
            "baseline_sparse_raw": sparse_raw,
            "baseline_sparse_rank50": sparse_rank50,
            "diagnostic_reconstruction_checks": checks,
        }
        all_rows.extend(rows.tolist())
    all_rows_arr = np.asarray(all_rows, dtype=np.int64)
    if (len(all_rows_arr) != EXPECTED_PRIMARY_ROWS or len(np.unique(all_rows_arr)) != EXPECTED_PRIMARY_ROWS
            or set(all_rows_arr.tolist()) != set(diag_rows.tolist())):
        raise BatchIntegrityError("R22 six-window primary rows are not a disjoint exact diagnostic match")
    return output


def _pilot_expected_identity(cutoff: int, cycles: np.ndarray,
                             freeze_identity: Mapping[str, Any]) -> dict[str, Any]:
    rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12)).astype(np.int64)
    return {
        "card_id": CARD_ID,
        "cutoff": int(cutoff),
        "freeze_signature": json_sha256(freeze_identity),
        "score_rows": rows.tolist(),
    }


def _load_pilot_bundle(cutoff: int, frame: pd.DataFrame, cycles: np.ndarray,
                       freeze_identity: Mapping[str, Any]) -> dict[str, Any]:
    expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12)).astype(np.int64)
    path = PILOT_BUNDLE_DIR / f"R22_fit{cutoff}.npz"
    sidecar_path = path.with_suffix(".json")
    bundle, sidecar = read_bundle(path, _pilot_expected_identity(cutoff, cycles, freeze_identity))
    if not np.array_equal(np.asarray(bundle["score_rows"], dtype=np.int64), expected_rows):
        raise BatchIntegrityError(f"R22 pilot {cutoff} score rows differ from official Train")
    if not np.array_equal(np.asarray(bundle["score_record_ids"], dtype=str), frame.iloc[expected_rows]["record_id"].astype(str).to_numpy(dtype=str)):
        raise BatchIntegrityError(f"R22 pilot {cutoff} score record IDs differ from official Train")
    score_cycles = cycles[expected_rows]
    if not np.array_equal(np.asarray(bundle["audit__score_cycles"], dtype=np.int64), score_cycles):
        raise BatchIntegrityError(f"R22 pilot {cutoff} score cycles differ from official Train")
    baseline = np.asarray(bundle["baseline_prediction"], dtype=np.float64)
    raw = np.asarray(bundle["raw_candidate"], dtype=np.float64)
    missing = stream_missing_mask(frame.iloc[expected_rows]["signal_stream"].tolist())
    policies = postprocess_policies(raw, baseline, score_cycles, ~missing)
    if not np.array_equal(np.asarray(bundle["prediction"], dtype=np.float64), policies["R22_ORIGINAL"]):
        raise BatchIntegrityError(f"R22 pilot {cutoff} original policy does not reconstruct from raw scores")
    if not np.array_equal(np.asarray(bundle["audit__candidate_rank50"], dtype=np.float64), policies["original_component"]):
        raise BatchIntegrityError(f"R22 pilot {cutoff} rank50 component does not reconstruct")
    if not np.array_equal(np.asarray(bundle["control__control_rows"], dtype=np.int64), expected_rows):
        raise BatchIntegrityError(f"R22 pilot {cutoff} control rows are misaligned")
    return {
        "cutoff": cutoff,
        "rows": expected_rows,
        "record_ids": np.asarray(bundle["score_record_ids"], dtype=str),
        "cycle_codes": score_cycles,
        "baseline_prediction": baseline,
        "raw_candidate": raw,
        "stream_present": ~missing,
        "policies": policies,
        "fit_state": sidecar["metadata"]["state"],
        "fit_metadata": sidecar["metadata"]["fit_metadata"],
        "npz_sha256": file_sha256(path),
        "sidecar_sha256": file_sha256(sidecar_path),
    }


def _validate_pilot_inputs(frame: pd.DataFrame, cycles: np.ndarray,
                           primary_controls: Mapping[int, Mapping[str, Any]],
                           pilots: Mapping[int, Mapping[str, Any]],
                           diagnostic: Mapping[str, np.ndarray]) -> None:
    diag_rows = np.asarray(diagnostic["h12_rows"], dtype=np.int64)
    diag_lookup = {int(row): i for i, row in enumerate(diag_rows)}
    for cutoff in (55, 67):
        pilot = pilots[cutoff]
        comp = primary_controls[cutoff]
        # Compare the reused pilot baseline and row identities against both
        # accepted cache reconstruction and the prior diagnostic artifact.
        positions = {int(row): i for i, row in enumerate(comp["rows"])}
        pilot_mask = np.isin(pilot["rows"], comp["rows"])
        p_rows = pilot["rows"][pilot_mask]
        pc = np.asarray([positions[int(row)] for row in p_rows], dtype=np.int64)
        if not np.array_equal(pilot["baseline_prediction"][pilot_mask], comp["baseline_prediction"][pc]):
            raise BatchIntegrityError(f"R22 pilot {cutoff} champion differs from cache reconstruction")
        if not np.array_equal(pilot["rows"][pilot_mask], comp["rows"][pc]):
            raise BatchIntegrityError(f"R22 pilot {cutoff} primary rows differ from cache reconstruction")
        try:
            dpos = np.asarray([diag_lookup[int(row)] for row in pilot["rows"]], dtype=np.int64)
        except KeyError as exc:
            raise BatchIntegrityError(f"R22 H12 diagnostic omits pilot row {exc.args[0]}") from exc
        if not np.array_equal(pilot["baseline_prediction"], np.asarray(diagnostic["h12_prediction"])[dpos]):
            raise BatchIntegrityError(f"R22 pilot {cutoff} champion differs from H12 diagnostic")
        if not np.array_equal(diag_rows[dpos], pilot["rows"]):
            raise BatchIntegrityError(f"R22 pilot {cutoff} H12 row order differs")
        expected_ids = frame.iloc[pilot["rows"]]["record_id"].astype(str).to_numpy(dtype=str)
        if not np.array_equal(pilot["record_ids"], expected_ids):
            raise BatchIntegrityError(f"R22 pilot {cutoff} IDs changed against official Train")


def _make_recipe(card: Mapping[str, Any], resource: Mapping[str, Any]) -> dict[str, Any]:
    from src.models.optimization_v3_batch1_r22 import XGB_PARAMETERS
    from src.models.regression_research_text import _SVD_PARAMS, _TFIDF_PARAMS

    return {
        "schema_version": 1,
        "experiment_id": "optimization_v3_final_continuation_R22",
        "card_id": CARD_ID,
        "decision_source": str(DECISION_PATH.relative_to(ROOT)).replace("\\", "/"),
        "training_recipe": {
            "inherited_model_class": "src.models.optimization_v3_batch1_r22.XGBoostPseudoHuberSVD32Regressor",
            "continuation_adapter": "src.models.optimization_v3_final_continuation_r22.FinalContinuationR22Regressor",
            "adapter_change": "admit only cutoffs 59/63/71/75 by initializing the unchanged parent and setting the validated cutoff; fit/predict code and all estimator/transforms unchanged",
            "estimator_parameters": dict(XGB_PARAMETERS),
            "feature_names": list(FEATURE_NAMES),
            "tfidf_parameters": {k: list(v) if isinstance(v, tuple) else (str(v) if k == "dtype" else v) for k, v in _TFIDF_PARAMS.items()},
            "svd_parameters": dict(_SVD_PARAMS),
            "svd_components_requested": 32,
            "structured_preprocessing": "inherited R22 fit-only numeric median/imputation, scaling, and categorical one-hot transforms",
            "missing_stream_handling": "inherited _text_values normalization and explicit missing indicator; stream-missing ranks retain full-cycle raw-score denominator for PCT_PRESENT and use exact champion output",
            "early_stopping": False,
            "validation_labels_passed_to_fit": False,
        },
        "new_cutoffs": list(CONTINUATION_CUTOFFS),
        "predictor_fits": len(CONTINUATION_CUTOFFS),
        "postprocessing_policies": {
            "R22_ORIGINAL": "clip(.75*champion + .25*clip(.5*raw + .5*(100*average_rank(raw within complete cycle)/cycle_n),0,100),0,100), all rows",
            "R22_PCT_PRESENT": "clip(.75*champion + .25*(100*average_rank(raw within complete cycle)/cycle_n),0,100) for stream-present; exact champion for missing; rank all rows",
        },
        "views": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "primary_rows": EXPECTED_PRIMARY_ROWS,
        "late_rows_W5_W6": EXPECTED_LATE_ROWS,
        "metric": "MAE; gain=champion_MAE-candidate_MAE; positive means candidate improves",
        "qualification": {
            "primary_gain_minimum": 0.03,
            "H12_harm_maximum": 0.05,
            "nonnegative_windows_minimum": 4,
            "late_W5_W6_gain_strictly_positive": True,
            "group_support_minimum": MIN_GROUP_SUPPORT,
            "group_MAE_harm_maximum": MAX_GROUP_HARM,
            "pct_present_additional_primary_MAE_no_worse_than_original": True,
            "pct_present_additional_late_gain_vs_original_minimum": 0.03,
        },
        "memory_policy": {"threads": THREADS, "job_cap_bytes": JOB_CAP_BYTES,
                          "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
                          "resource_preflight": dict(resource)},
        "pilot_origins_reused": [55, 67],
        "official_test_access": False,
        "kaggle_access": False,
        "queue_card": dict(card),
    }


def prepare_context() -> dict[str, Any]:
    queue = r22_pilot.load_queue(r22_pilot.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    train_path, frame, target, cycles = r22_pilot._read_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train changed from frozen R22 source")
    if frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Train record IDs are duplicated")
    base, sparse_pilot, fullsix, diagnostic = _control_sources()
    hashes = {
        "paired_lgb_ridge_cache": file_sha256(r01.BASE_CACHE),
        "sparse_pilot_cache": file_sha256(r01.SPARSE_CACHE),
        "sparse_fullsix_cache": file_sha256(FULLSIX_CACHE),
        "champion_diagnostic_cache": file_sha256(r01.DIAGNOSTIC_CACHE),
    }
    expected_hashes = {
        "paired_lgb_ridge_cache": r01.EXPECTED_BASE_CACHE_SHA256,
        "sparse_pilot_cache": r01.EXPECTED_SPARSE_CACHE_SHA256,
        "sparse_fullsix_cache": FULLSIX_CACHE_SHA256,
        "champion_diagnostic_cache": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
    }
    if hashes != expected_hashes:
        raise BatchIntegrityError(f"R22 accepted control cache hash mismatch: {hashes}")
    if _read_json(FULLSIX_REPORT).get("all_six_primary_windows_and_both_H12_views_completed") is not True:
        raise BatchIntegrityError("Accepted full-six control report does not attest full-six/H12 completion")
    if _read_json(FULLSIX_REPORT).get("test_access") is not False:
        raise BatchIntegrityError("Accepted full-six control report does not affirm no Test access")

    primary_controls = _build_primary_controls(frame, cycles, diagnostic, base, sparse_pilot, fullsix)
    pilot_freeze = _read_json(PILOT_FREEZE)
    pilot_identity = pilot_freeze["identity"]
    if json_sha256(pilot_identity) != pilot_freeze.get("signature"):
        raise BatchIntegrityError("Original R22 pilot freeze signature does not verify")
    # The shared queue's whole-file hash changed when unrelated R10/R13
    # result fields were updated. Bind the current queue below, and fail closed
    # unless the R22 queue leaf and its Markdown card still match the pilot.
    if (card != pilot_identity.get("card")
            or file_sha256(CARD_PATH) != r22_pilot.CARD_SHA256
            or file_sha256(r01.ACCEPTED_MANIFEST) != r22_pilot.MANIFEST_SHA256):
        raise BatchIntegrityError("Current R22 queue card or card document differs from its frozen pilot recipe")
    for key, expected in r22_pilot.EXPECTED_CARD.items():
        if card.get(key) != expected:
            raise BatchIntegrityError(f"R22 card changed at frozen field {key}")
    baseline = queue["baseline"]["regression"]
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != r22_pilot.MANIFEST_SHA256
            or float(baseline.get("primary", float("nan"))) != 11.559390078665364
            or float(baseline.get("H12", float("nan"))) != 14.04753258299653):
        raise BatchIntegrityError("Current queue champion identity/metrics differ from the R22 frozen baseline")
    for key in r22_pilot.EXPECTED_GATE_KEYS:
        if key not in queue["evaluation"]["regression_gate"] or not np.isfinite(float(queue["evaluation"]["regression_gate"][key])):
            raise BatchIntegrityError(f"Current queue regression gate is missing/invalid at {key}")
    pilot_hashes: dict[str, str] = {}
    pilots: dict[int, dict[str, Any]] = {}
    for cutoff in (55, 67):
        npz_path = PILOT_BUNDLE_DIR / f"R22_fit{cutoff}.npz"
        sidecar_path = npz_path.with_suffix(".json")
        for name, path in (("npz", npz_path), ("json", sidecar_path)):
            observed = file_sha256(path)
            expected = PILOT_VERIFIED[str(cutoff)][name]
            if observed != expected:
                raise BatchIntegrityError(f"Frozen R22 pilot {cutoff} {name} hash changed")
            pilot_hashes[f"R22_fit{cutoff}.{name}"] = observed
        pilots[cutoff] = _load_pilot_bundle(cutoff, frame, cycles, pilot_identity)
    _validate_pilot_inputs(frame, cycles, primary_controls, pilots, diagnostic)

    fit_rows = {str(cutoff): int(np.sum(cycles <= cutoff)) for cutoff in CONTINUATION_CUTOFFS}
    resources = {cutoff: _resource_preflight(fit_rows[str(cutoff)]) for cutoff in CONTINUATION_CUTOFFS}
    current_free = int(psutil.virtual_memory().available)
    live_lock_check = _check_lock_available()
    budget_snapshot = _budget_snapshot()
    if any(item["status"] != "PASS" for item in resources.values()):
        raise BatchIntegrityError(f"R22 static resource estimate is blocked: {resources}")
    if not live_lock_check["available"]:
        raise BatchIntegrityError("Global V3 training lock is currently held")
    if budget_snapshot["reserved_total"] + 8 > MAX_SUCCESSFUL_FITS:
        raise BatchIntegrityError(f"Joint eight-fit reservation exceeds remaining cap: {budget_snapshot}")
    if PREFLIGHT_PATH.is_file():
        old_preflight = _read_json(PREFLIGHT_PATH)
        source_head = str(old_preflight.get("source_head_before_fit") or _git_head())
        budget_before_reservation = old_preflight.get("global_budget_snapshot", budget_snapshot)
    else:
        source_head = _git_head()
        budget_before_reservation = budget_snapshot

    input_files = {
        "official_regression_train": train_path,
        "canonical_queue": r22_pilot.QUEUE_PATH,
        "canonical_R22_card": CARD_PATH,
        "accepted_regression_manifest": r01.ACCEPTED_MANIFEST,
        "paired_lgb_ridge_cache": r01.BASE_CACHE,
        "sparse_pilot_cache": r01.SPARSE_CACHE,
        "sparse_fullsix_cache": FULLSIX_CACHE,
        "accepted_fullsix_report": FULLSIX_REPORT,
        "champion_diagnostic_cache": r01.DIAGNOSTIC_CACHE,
        "regression_diagnostics_report": r01.DIAGNOSTICS_REPORT,
        "v2_regression_findings": r01.FINDINGS_REPORT,
        "optimization_v3_feasibility": ROOT / "reports/optimization_v3/research_backlog/feasibility.json",
        "R22_pilot_freeze": PILOT_FREEZE,
        "R22_pilot55_npz": PILOT_BUNDLE_DIR / "R22_fit55.npz",
        "R22_pilot55_sidecar": PILOT_BUNDLE_DIR / "R22_fit55.json",
        "R22_pilot67_npz": PILOT_BUNDLE_DIR / "R22_fit67.npz",
        "R22_pilot67_sidecar": PILOT_BUNDLE_DIR / "R22_fit67.json",
        "decision": DECISION_PATH,
        "prompt": PROMPT_PATH,
        "guard_audit": GUARD_AUDIT_PATH,
    }
    recipe = _make_recipe(card, _resource_identity(resources))
    recipe_payload = json.dumps(recipe, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    identity_seed = {
        "schema_version": 1,
        "experiment_id": "optimization_v3_final_continuation_R22",
        "card": card,
        "feature_names": list(FEATURE_NAMES),
        "input_sha256": {name: file_sha256(path) for name, path in sorted(input_files.items())},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in SOURCE_PATHS},
        "evaluation_views": {"primary": {name: list(bounds) for name, bounds in WINDOWS.items()},
                             "H12": {"55": [56, 67], "67": [68, 79]},
                             "qualification_views": list(REQUIRED_VIEWS)},
        "fit_cutoffs": list(CONTINUATION_CUTOFFS),
        "reused_pilot_artifacts": pilot_hashes,
        "control_cache_hashes": hashes,
        "resource_policy": _resource_identity(resources),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
        "kaggle_access": False,
    }
    from src.models.optimization_v3_batch1 import package_versions
    identity_seed["package_versions"] = package_versions([
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "threadpoolctl", "psutil"
    ])
    identity_seed["recipe_sha256"] = hashlib.sha256(recipe_payload).hexdigest()
    return {
        "queue": queue,
        "card": card,
        "train_path": Path(train_path),
        "frame": frame,
        "target": target,
        "cycles": cycles,
        "primary_controls": primary_controls,
        "pilots": pilots,
        "diagnostic": diagnostic,
        "input_files": input_files,
        "input_hashes": identity_seed["input_sha256"],
        "source_hashes": identity_seed["source_sha256"],
        "control_cache_hashes": hashes,
        "pilot_hashes": pilot_hashes,
        "resources": resources,
        "current_free_ram_bytes": current_free,
        "source_head": source_head,
        "lock_check": live_lock_check,
        "budget_snapshot": budget_snapshot,
        "budget_snapshot_pre_reservation": budget_before_reservation,
        "identity": identity_seed,
        "signature": json_sha256(identity_seed),
        "recipe": recipe,
    }


def _check_lock_available() -> dict[str, Any]:
    """Safely contend for, then release, the shared OS lock without deleting it."""
    from src.models.optimization_v3_batch1 import exclusive_training_slot
    try:
        with exclusive_training_slot(TRAINING_LOCK):
            return {"available": True, "lock_path": str(TRAINING_LOCK.relative_to(ROOT)),
                    "test": "OS-level nonblocking lock acquisition and release", "file_deleted": False}
    except BatchIntegrityError:
        return {"available": False, "lock_path": str(TRAINING_LOCK.relative_to(ROOT)),
                "test": "OS-level nonblocking lock acquisition", "file_deleted": False}


def _budget_snapshot() -> dict[str, Any]:
    reservations: dict[str, tuple[int, str | None]] = {}
    lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines() if LEDGER_PATH.is_file() else []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "BUDGET_RESERVED":
            continue
        key = str(event["budget_key"])
        reservation = (int(event["predictor_fits"]), str(event.get("card_id")) if event.get("card_id") is not None else None)
        if key in reservations:
            raise BatchIntegrityError(f"Duplicate budget reservation key at ledger line {line_number}: {key}")
        reservations[key] = reservation
    return {
        "reservation_count": len(reservations),
        "reserved_total": sum(item[0] for item in reservations.values()),
        "remaining_before_joint_reservation": MAX_SUCCESSFUL_FITS - sum(item[0] for item in reservations.values()),
        "joint_reservation_present": JOINT_RESERVATION_KEY in reservations,
        "joint_reservation": reservations.get(JOINT_RESERVATION_KEY),
        "ledger_sha256": file_sha256(LEDGER_PATH) if LEDGER_PATH.is_file() else None,
    }


def _preflight_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "READY_FOR_FREEZE" if all(x["status"] == "PASS" for x in context["resources"].values()) else "BLOCKED",
        "scope": "R22 only; four new cutoffs and one fixed postprocessing comparator",
        "new_fit_count": 4,
        "new_cutoffs": list(CONTINUATION_CUTOFFS),
        "reused_pilot_cutoffs": [55, 67],
        "official_train_sha256": file_sha256(context["train_path"]),
        "input_hashes": context["input_hashes"],
        "source_hashes": context["source_hashes"],
        "control_cache_hashes": context["control_cache_hashes"],
        "pilot_bundle_hashes": context["pilot_hashes"],
        "primary_control_rows": int(sum(len(v["rows"]) for v in context["primary_controls"].values())),
        "primary_control_rows_unique": int(len(set(np.concatenate([v["rows"] for v in context["primary_controls"].values()]).tolist()))),
        "primary_control_reconstruction": "exact equality to accepted oof_residuals.npz components and primary_prediction on all six windows",
        "resources": context["resources"],
        "available_physical_ram_bytes_at_preflight": context["current_free_ram_bytes"],
        "source_head_before_fit": context["source_head"],
        "global_training_lock": context["lock_check"],
        "global_budget_snapshot": context["budget_snapshot"],
        "atomic_joint_reservation": {"budget_key": JOINT_RESERVATION_KEY, "card_id": JOINT_RESERVATION_CARD,
                                      "predictor_fits": 8, "cap": MAX_SUCCESSFUL_FITS,
                                      "reserved_before_fit": True},
        "source_data_boundary": {"official_train_only": True, "official_test_opened": False,
                                  "sample_submission_opened": False, "kaggle_access": False},
    }


def freeze_context(context: Mapping[str, Any]) -> dict[str, Any]:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = dict(context["recipe"])
    if RECIPE_PATH.is_file():
        if _read_json(RECIPE_PATH) != recipe:
            raise BatchIntegrityError("Existing R22 continuation recipe differs; refusing overwrite")
    else:
        write_json_exclusive(RECIPE_PATH, recipe)
    identity = dict(context["identity"])
    if identity.get("recipe_sha256") != file_sha256(RECIPE_PATH):
        raise BatchIntegrityError("R22 canonical recipe hash differs from preflight identity")
    signature = ensure_frozen(FREEZE_PATH, identity)
    if signature != context["signature"]:
        raise BatchIntegrityError("R22 freeze signature differs from prepared context")
    payload = _preflight_payload(context)
    payload.update({
        "status": "FROZEN_READY",
        "recipe_path": str(RECIPE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "recipe_sha256": file_sha256(RECIPE_PATH),
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "freeze_signature": signature,
        "candidate_outputs_or_metrics_computed": False,
    })
    if PREFLIGHT_PATH.exists():
        existing_preflight = _read_json(PREFLIGHT_PATH)
        if (existing_preflight.get("freeze_signature") != signature
                or existing_preflight.get("recipe_sha256") != payload["recipe_sha256"]):
            raise BatchIntegrityError("Existing R22 preflight receipt differs from frozen identity")
        payload = existing_preflight
    else:
        write_json_exclusive(PREFLIGHT_PATH, payload)
    return payload


def _resource_recheck(context: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    fit_rows = int(np.sum(np.asarray(context["cycles"], dtype=np.int64) <= int(cutoff)))
    available = int(psutil.virtual_memory().available)
    status = _resource_preflight(fit_rows, available)
    process = psutil.Process(os.getpid())
    status["runner_rss_before_fit_bytes"] = int(process.memory_info().rss)
    if status["status"] != "PASS":
        raise BatchIntegrityError(f"R22 resource preflight failed immediately before cutoff {cutoff}: {status}")
    return status


def _fit_event_state(cutoff: int) -> dict[str, bool]:
    state = {"started": False, "saved": False, "failed": False}
    if not LEDGER_PATH.is_file():
        return state
    for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("experiment_id") != "optimization_v3_final_continuation_R22" or int(event.get("cutoff", -1)) != int(cutoff):
            continue
        kind = event.get("event")
        if kind == "CONTINUATION_FIT_STARTED":
            state["started"] = True
        elif kind == "CONTINUATION_FIT_SAVED":
            state["saved"] = True
        elif kind == "CONTINUATION_FIT_FAILED":
            state["failed"] = True
    return state


def _bundle_identity(context: Mapping[str, Any], cutoff: int, rows: np.ndarray) -> dict[str, Any]:
    return {
        "experiment_id": "optimization_v3_final_continuation_R22",
        "card_id": CARD_ID,
        "cutoff": int(cutoff),
        "freeze_signature": str(context["signature"]),
        "score_rows": np.asarray(rows, dtype=np.int64).tolist(),
    }


def _validate_new_bundle(context: Mapping[str, Any], cutoff: int,
                         bundle_path: Path) -> dict[str, Any]:
    low, high = cutoff + 1, cutoff + 4
    rows = np.flatnonzero((context["cycles"] >= low) & (context["cycles"] <= high)).astype(np.int64)
    identity = _bundle_identity(context, cutoff, rows)
    arrays, sidecar = read_bundle(bundle_path, identity)
    for key in ("score_rows", "score_record_ids", "score_cycle_codes", "champion_prediction",
                "raw_candidate", "cycle_percentile", "original_component", "R22_ORIGINAL",
                "R22_PCT_PRESENT", "stream_present"):
        if key not in arrays:
            raise BatchIntegrityError(f"R22 cutoff {cutoff} bundle lacks {key}")
    frame = context["frame"]
    cycles = np.asarray(context["cycles"])[rows]
    if not np.array_equal(arrays["score_rows"], rows):
        raise BatchIntegrityError(f"R22 saved cutoff {cutoff} rows differ from official Train")
    expected_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
    if not np.array_equal(np.asarray(arrays["score_record_ids"], dtype=str), expected_ids):
        raise BatchIntegrityError(f"R22 saved cutoff {cutoff} IDs differ from official Train")
    if not np.array_equal(np.asarray(arrays["score_cycle_codes"], dtype=np.int64), cycles):
        raise BatchIntegrityError(f"R22 saved cutoff {cutoff} cycles differ from official Train")
    expected_control = context["primary_controls"][cutoff]
    if not np.array_equal(arrays["champion_prediction"], expected_control["baseline_prediction"]):
        raise BatchIntegrityError(f"R22 saved cutoff {cutoff} champion control changed")
    missing = stream_missing_mask(frame.iloc[rows]["signal_stream"].tolist())
    if not np.array_equal(np.asarray(arrays["stream_present"], dtype=bool), ~missing):
        raise BatchIntegrityError(f"R22 saved cutoff {cutoff} stream availability changed")
    # Independent verification intentionally reconstructs with scipy rankdata,
    # not the fit adapter's postprocessing function.
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    expected_pct = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        expected_pct[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    component = np.clip(0.5 * raw + 0.5 * expected_pct, 0.0, 100.0)
    original = np.clip(0.75 * expected_control["baseline_prediction"] + 0.25 * component, 0.0, 100.0)
    pct_present_value = np.clip(0.75 * expected_control["baseline_prediction"] + 0.25 * expected_pct, 0.0, 100.0)
    pct_present = np.where(~missing, pct_present_value, expected_control["baseline_prediction"])
    comparisons = {
        "percentile": np.array_equal(np.asarray(arrays["cycle_percentile"], dtype=np.float64), expected_pct),
        "original_component": np.array_equal(np.asarray(arrays["original_component"], dtype=np.float64), component),
        "R22_ORIGINAL": np.array_equal(np.asarray(arrays["R22_ORIGINAL"], dtype=np.float64), original),
        "R22_PCT_PRESENT": np.array_equal(np.asarray(arrays["R22_PCT_PRESENT"], dtype=np.float64), pct_present),
    }
    if not all(comparisons.values()):
        raise BatchIntegrityError(f"R22 independent policy reconstruction failed at cutoff {cutoff}: {comparisons}")
    metadata = sidecar["metadata"]
    if metadata.get("fit_predictor_count") != 1 or metadata.get("fit_metadata", {}).get("score_labels_passed_to_fit") is not False:
        raise BatchIntegrityError(f"R22 cutoff {cutoff} saved fit metadata violates one-fit/no-label contract")
    return {"arrays": arrays, "sidecar": sidecar, "policy_reconstruction": comparisons}


def _load_or_fit_cutoff(context: dict[str, Any], cutoff: int,
                        *, lock_held: bool) -> dict[str, Any]:
    if not lock_held:
        raise BatchIntegrityError("R22 continuation cutoff must run under the global training lock")
    rows = context["primary_controls"][cutoff]["rows"]
    npz_path = RUN_DIR / f"R22_fit{cutoff}.npz"
    sidecar_path = npz_path.with_suffix(".json")
    if npz_path.exists() or sidecar_path.exists():
        if not npz_path.exists() or not sidecar_path.exists():
            raise BatchIntegrityError(f"R22 cutoff {cutoff} has a partial bundle pair; refusing overwrite/refit")
        checked = _validate_new_bundle(context, cutoff, npz_path)
        return {"status": "RESUMED_VERIFIED", "cutoff": cutoff,
                "bundle_sha256": file_sha256(npz_path), "sidecar_sha256": file_sha256(sidecar_path),
                "fit_seconds": checked["sidecar"]["metadata"].get("fit_seconds"),
                "policy_reconstruction": checked["policy_reconstruction"]}

    prior = _fit_event_state(cutoff)
    if prior["started"]:
        raise BatchIntegrityError(f"R22 cutoff {cutoff} has a prior started attempt without a complete bundle; manual review required")
    resource = _resource_recheck(context, cutoff)
    fit_rows = np.flatnonzero(context["cycles"] <= cutoff).astype(np.int64)
    fit_features = context["frame"].iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    score_features = context["frame"].iloc[rows].loc[:, list(FEATURE_NAMES)].copy()
    fit_target = np.asarray(context["target"], dtype=np.float64)[fit_rows]
    if not np.array_equal(fit_features.index.to_numpy(dtype=np.int64), fit_rows):
        raise BatchIntegrityError(f"R22 cutoff {cutoff} fit prefix row positions changed")
    if not np.array_equal(score_features.index.to_numpy(dtype=np.int64), rows):
        raise BatchIntegrityError(f"R22 cutoff {cutoff} score row positions changed")
    if np.any(context["cycles"][fit_rows] > cutoff) or np.any(context["cycles"][rows] <= cutoff):
        raise BatchIntegrityError(f"R22 cutoff {cutoff} temporal boundary is invalid")

    started_at = _utc_now()
    append_event(LEDGER_PATH, {
        "event": "CONTINUATION_FIT_STARTED",
        "at_utc": started_at,
        "experiment_id": "optimization_v3_final_continuation_R22",
        "card_id": CARD_ID,
        "cutoff": int(cutoff),
        "predictor_fit_count": 1,
        "freeze_signature": context["signature"],
        "fit_rows": int(len(fit_rows)),
        "score_rows": int(len(rows)),
        "resource_preflight": resource,
        "scope_amendment_signature": context.get("scope_amendment_signature"),
    })
    start = time.perf_counter()
    try:
        result = fit_raw_predict(fit_features, fit_target, score_features, cutoff=cutoff)
        fit_seconds = float(time.perf_counter() - start)
        raw = np.asarray(result["raw_prediction"], dtype=np.float64)
        control = context["primary_controls"][cutoff]
        frame = context["frame"]
        cycles = np.asarray(context["cycles"])[rows]
        missing = stream_missing_mask(frame.iloc[rows]["signal_stream"].tolist())
        present = ~missing
        policies = postprocess_policies(raw, control["baseline_prediction"], cycles, present)
        arrays = {
            "score_rows": rows.astype(np.int64),
            "score_record_ids": control["record_ids"].astype(str),
            "score_cycle_codes": cycles.astype(np.int64),
            "champion_prediction": np.asarray(control["baseline_prediction"], dtype=np.float64),
            "baseline_lgb_svd32": np.asarray(control["baseline_lgb_svd32"], dtype=np.float64),
            "baseline_hybrid_ridge": np.asarray(control["baseline_hybrid_ridge"], dtype=np.float64),
            "baseline_sparse_raw": np.asarray(control["baseline_sparse_raw"], dtype=np.float64),
            "baseline_sparse_rank50": np.asarray(control["baseline_sparse_rank50"], dtype=np.float64),
            "raw_candidate": raw,
            "cycle_percentile": policies["cycle_percentile"],
            "original_component": policies["original_component"],
            "R22_ORIGINAL": policies["R22_ORIGINAL"],
            "R22_PCT_PRESENT": policies["R22_PCT_PRESENT"],
            "stream_present": present.astype(np.bool_),
        }
        metadata = {
            "experiment_id": "optimization_v3_final_continuation_R22",
            "card_id": CARD_ID,
            "cutoff": int(cutoff),
            "fit_predictor_count": 1,
            "fit_seconds": fit_seconds,
            "fit_metadata": result["fit_metadata"],
            "state": result["state"],
            "resource_preflight": resource,
            "rss_after_fit_bytes": int(psutil.Process(os.getpid()).memory_info().rss),
            "score_labels_passed_to_fit": False,
            "official_test_used": False,
        }
        identity = _bundle_identity(context, cutoff, rows)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = write_bundle_exclusive(npz_path, arrays, identity, metadata)
        checked = _validate_new_bundle(context, cutoff, npz_path)
        append_event(LEDGER_PATH, {
            "event": "CONTINUATION_FIT_SAVED",
            "at_utc": _utc_now(),
            "experiment_id": "optimization_v3_final_continuation_R22",
            "card_id": CARD_ID,
            "cutoff": int(cutoff),
            "predictor_fit_count": 1,
            "fit_seconds": fit_seconds,
            "freeze_signature": context["signature"],
            "npz_sha256": file_sha256(npz_path),
            "sidecar_sha256": file_sha256(sidecar_path),
            "scope_amendment_signature": context.get("scope_amendment_signature"),
            "policy_reconstruction": checked["policy_reconstruction"],
        })
        return {"status": "FIT_SAVED_VERIFIED", "cutoff": cutoff,
                "bundle_sha256": file_sha256(npz_path), "sidecar_sha256": file_sha256(sidecar_path),
                "fit_seconds": fit_seconds, "rss_after_fit_bytes": metadata["rss_after_fit_bytes"],
                "policy_reconstruction": checked["policy_reconstruction"]}
    except Exception as exc:
        append_event(LEDGER_PATH, {
            "event": "CONTINUATION_FIT_FAILED",
            "at_utc": _utc_now(),
            "experiment_id": "optimization_v3_final_continuation_R22",
            "card_id": CARD_ID,
            "cutoff": int(cutoff),
            "predictor_fit_count": 1,
            "freeze_signature": context["signature"],
            "scope_amendment_signature": context.get("scope_amendment_signature"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


def execute_cutoff(context: dict[str, Any], freeze_signature: str, cutoff: int,
                   *, lock_held: bool) -> dict[str, Any]:
    if freeze_signature != context["signature"]:
        raise BatchIntegrityError("R22 continuation freeze signature mismatch")
    frozen = _read_json(FREEZE_PATH)
    if frozen.get("identity") != context["identity"] or frozen.get("signature") != freeze_signature:
        raise BatchIntegrityError("R22 continuation freeze differs from current source/input identity")
    cutoff = int(cutoff)
    if cutoff not in CONTINUATION_CUTOFFS:
        raise BatchIntegrityError(f"R22 continuation cutoff not in scope: {cutoff}")
    return _load_or_fit_cutoff(context, cutoff, lock_held=lock_held)


def _pilot_as_fold(pilot: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "rows": np.asarray(pilot["rows"], dtype=np.int64),
        "cycle_codes": np.asarray(pilot["cycle_codes"], dtype=np.int64),
        "record_ids": np.asarray(pilot["record_ids"], dtype=str),
        "baseline_prediction": np.asarray(pilot["baseline_prediction"], dtype=np.float64),
        "raw_candidate": np.asarray(pilot["raw_candidate"], dtype=np.float64),
        "stream_present": np.asarray(pilot["stream_present"], dtype=bool),
        **{k: np.asarray(v, dtype=np.float64) for k, v in pilot["policies"].items()},
    }


def load_all_folds(context: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    folds: dict[int, dict[str, np.ndarray]] = {c: _pilot_as_fold(context["pilots"][c]) for c in (55, 67)}
    for cutoff in CONTINUATION_CUTOFFS:
        npz_path = RUN_DIR / f"R22_fit{cutoff}.npz"
        checked = _validate_new_bundle(context, cutoff, npz_path)
        arrays = checked["arrays"]
        folds[cutoff] = {
            "rows": np.asarray(arrays["score_rows"], dtype=np.int64),
            "cycle_codes": np.asarray(arrays["score_cycle_codes"], dtype=np.int64),
            "record_ids": np.asarray(arrays["score_record_ids"], dtype=str),
            "baseline_prediction": np.asarray(arrays["champion_prediction"], dtype=np.float64),
            "raw_candidate": np.asarray(arrays["raw_candidate"], dtype=np.float64),
            "stream_present": np.asarray(arrays["stream_present"], dtype=bool),
            "cycle_percentile": np.asarray(arrays["cycle_percentile"], dtype=np.float64),
            "original_component": np.asarray(arrays["original_component"], dtype=np.float64),
            "R22_ORIGINAL": np.asarray(arrays["R22_ORIGINAL"], dtype=np.float64),
            "R22_PCT_PRESENT": np.asarray(arrays["R22_PCT_PRESENT"], dtype=np.float64),
        }
    return folds


def _view_arrays(context: Mapping[str, Any], folds: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    cycles_all = np.asarray(context["cycles"], dtype=np.int64)
    views: dict[str, dict[str, np.ndarray]] = {}
    window_rows: list[np.ndarray] = []
    for name, (cutoff, low, high) in WINDOWS.items():
        fold = folds[cutoff]
        mask = (fold["cycle_codes"] >= low) & (fold["cycle_codes"] <= high)
        rows = np.asarray(fold["rows"], dtype=np.int64)[mask]
        if not np.array_equal(rows, np.flatnonzero((cycles_all >= low) & (cycles_all <= high))):
            raise BatchIntegrityError(f"R22 {name} row denominator/order differs from official Train")
        views[name] = {"rows": rows,
                       "baseline": np.asarray(fold["baseline_prediction"])[mask],
                       "R22_ORIGINAL": np.asarray(fold["R22_ORIGINAL"])[mask],
                       "R22_PCT_PRESENT": np.asarray(fold["R22_PCT_PRESENT"])[mask]}
        window_rows.append(rows)
    primary_rows = np.concatenate(window_rows)
    if len(primary_rows) != EXPECTED_PRIMARY_ROWS or len(np.unique(primary_rows)) != EXPECTED_PRIMARY_ROWS:
        raise BatchIntegrityError("R22 primary windows are not disjoint 38,588 rows")
    views["primary"] = {key: np.concatenate([views[w][key] for w in WINDOWS])
                         for key in ("rows", "baseline", "R22_ORIGINAL", "R22_PCT_PRESENT")}
    # H12 is independently assembled from the complete reused pilot55/67 outputs.
    early = folds[55]
    late = folds[67]
    h12_rows = np.concatenate([early["rows"], late["rows"]]).astype(np.int64)
    h12_cycles = cycles_all[h12_rows]
    if (len(h12_rows) != EXPECTED_PRIMARY_ROWS or len(np.unique(h12_rows)) != EXPECTED_PRIMARY_ROWS
            or set(h12_rows.tolist()) != set(primary_rows.tolist())):
        raise BatchIntegrityError("R22 H12 pilot reuse does not cover the exact full primary row set")
    views["H12"] = {
        "rows": h12_rows,
        "baseline": np.concatenate([early["baseline_prediction"], late["baseline_prediction"]]),
        "R22_ORIGINAL": np.concatenate([early["R22_ORIGINAL"], late["R22_ORIGINAL"]]),
        "R22_PCT_PRESENT": np.concatenate([early["R22_PCT_PRESENT"], late["R22_PCT_PRESENT"]]),
    }
    for label, low, high in (("H12_E", 56, 67), ("H12_L", 68, 79)):
        mask = (h12_cycles >= low) & (h12_cycles <= high)
        views[label] = {key: np.asarray(value)[mask] for key, value in views["H12"].items()}
    return views


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 1 or len(actual) == 0:
        raise BatchIntegrityError("R22 MAE inputs are empty or misaligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise BatchIntegrityError("R22 MAE inputs contain nonfinite values")
    return float(np.mean(np.abs(actual - predicted)))


def _group_table(frame: pd.DataFrame, target: np.ndarray,
                 views: Mapping[str, Mapping[str, np.ndarray]], policy: str) -> dict[str, Any]:
    missing_all = stream_missing_mask(frame["signal_stream"].tolist())
    groups: dict[str, Any] = {"minimum_support": MIN_GROUP_SUPPORT, "maximum_allowed_harm": MAX_GROUP_HARM, "views": {}}
    for view_name in REQUIRED_VIEWS:
        view = views[view_name]
        rows = np.asarray(view["rows"], dtype=np.int64)
        actual = np.asarray(target, dtype=np.float64)[rows]
        candidate = np.asarray(view[policy], dtype=np.float64)
        baseline = np.asarray(view["baseline"], dtype=np.float64)
        masks = {
            "signal_stream_missing": missing_all[rows],
            "threat_rank_le_5": actual <= 5.0,
            "threat_rank_ge_95": actual >= 95.0,
        }
        groups["views"][view_name] = {}
        for label, mask in masks.items():
            n = int(mask.sum())
            supported = n >= MIN_GROUP_SUPPORT
            candidate_mae = _mae(actual[mask], candidate[mask]) if n else None
            baseline_mae = _mae(actual[mask], baseline[mask]) if n else None
            harm = float(candidate_mae - baseline_mae) if n else None
            groups["views"][view_name][label] = {
                "n": n,
                "supported": supported,
                "candidate_mae": candidate_mae,
                "champion_mae": baseline_mae,
                "harm": harm,
                "pass": (harm <= MAX_GROUP_HARM) if supported else None,
                "unsupported_is_descriptive_only": not supported,
            }
    return groups


def _policy_metrics(frame: pd.DataFrame, target: np.ndarray,
                    views: Mapping[str, Mapping[str, np.ndarray]], policy: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {"policy": policy, "views": {}}
    for view_name in REQUIRED_VIEWS:
        view = views[view_name]
        rows = np.asarray(view["rows"], dtype=np.int64)
        actual = np.asarray(target, dtype=np.float64)[rows]
        candidate = np.asarray(view[policy], dtype=np.float64)
        baseline = np.asarray(view["baseline"], dtype=np.float64)
        candidate_mae = _mae(actual, candidate)
        champion_mae = _mae(actual, baseline)
        metrics["views"][view_name] = {
            "n": int(len(rows)),
            "candidate_mae": candidate_mae,
            "champion_mae": champion_mae,
            "gain_vs_champion": float(champion_mae - candidate_mae),
            "harm_vs_champion": float(candidate_mae - champion_mae),
        }
    late_rows = np.concatenate([views["W5"]["rows"], views["W6"]["rows"]]).astype(np.int64)
    if len(np.unique(late_rows)) != EXPECTED_LATE_ROWS:
        raise BatchIntegrityError("R22 W5+W6 rows do not equal 15,953 unique records")
    late_candidate = np.concatenate([views["W5"][policy], views["W6"][policy]])
    late_baseline = np.concatenate([views["W5"]["baseline"], views["W6"]["baseline"]])
    late_target = np.asarray(target, dtype=np.float64)[late_rows]
    metrics["late_W5_W6"] = {
        "n": int(len(late_rows)),
        "candidate_mae": _mae(late_target, late_candidate),
        "champion_mae": _mae(late_target, late_baseline),
        "gain_vs_champion": float(_mae(late_target, late_baseline) - _mae(late_target, late_candidate)),
    }
    diagnostics: dict[str, Any] = {"H12_E": {}, "H12_L": {}}
    for view_name in ("H12_E", "H12_L"):
        view = views[view_name]
        rows = np.asarray(view["rows"], dtype=np.int64)
        actual = np.asarray(target, dtype=np.float64)[rows]
        candidate = np.asarray(view[policy], dtype=np.float64)
        baseline = np.asarray(view["baseline"], dtype=np.float64)
        diagnostics[view_name] = {
            "n": int(len(rows)),
            "candidate_mae": _mae(actual, candidate),
            "champion_mae": _mae(actual, baseline),
            "gain_vs_champion": float(_mae(actual, baseline) - _mae(actual, candidate)),
            "gate_applied": False,
        }
    metrics["diagnostics"] = diagnostics
    metrics["groups"] = _group_table(frame, target, views, policy)
    return metrics


def _qualification(metrics: Mapping[str, Any]) -> dict[str, Any]:
    m = metrics["views"]
    group_failures = []
    supported_checks = 0
    for view_name, groups in metrics["groups"]["views"].items():
        for group_name, item in groups.items():
            if item["supported"]:
                supported_checks += 1
                if item["harm"] > MAX_GROUP_HARM:
                    group_failures.append({"view": view_name, "group": group_name, "harm": item["harm"], "n": item["n"]})
    nonnegative_windows = sum(m[name]["gain_vs_champion"] >= 0.0 for name in ("W1", "W2", "W3", "W4", "W5", "W6"))
    gates = {
        "primary_gain_at_least_0_03": m["primary"]["gain_vs_champion"] >= 0.03,
        "H12_harm_at_most_0_05": m["H12"]["harm_vs_champion"] <= 0.05,
        "at_least_four_of_six_windows_nonnegative": nonnegative_windows >= 4,
        "W5_W6_gain_positive": metrics["late_W5_W6"]["gain_vs_champion"] > 0.0,
        "all_supported_group_harm_at_most_0_25": not group_failures,
    }
    return {
        "qualified": all(gates.values()),
        "gates": gates,
        "nonnegative_windows": int(nonnegative_windows),
        "supported_group_checks": supported_checks,
        "group_failures": group_failures,
    }


def _independent_recompute(context: Mapping[str, Any], folds: Mapping[int, Mapping[str, np.ndarray]],
                           views: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    # Recompute each primary output directly from saved raw scores using a
    # separately written rankdata path; then compare bundle values and metrics.
    checks: dict[str, Any] = {"folds": {}, "all_views": {}}
    for cutoff, fold in folds.items():
        raw = np.asarray(fold["raw_candidate"], dtype=np.float64)
        cycles = np.asarray(fold["cycle_codes"], dtype=np.int64)
        baseline = np.asarray(fold["baseline_prediction"], dtype=np.float64)
        present = np.asarray(fold["stream_present"], dtype=bool)
        percentile = np.empty(raw.shape, dtype=np.float64)
        for cycle in np.unique(cycles):
            mask = cycles == cycle
            percentile[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        original = np.clip(0.75 * baseline + 0.25 * np.clip(0.5 * raw + 0.5 * percentile, 0.0, 100.0), 0.0, 100.0)
        present_policy = np.where(present, np.clip(0.75 * baseline + 0.25 * percentile, 0.0, 100.0), baseline)
        checks["folds"][str(cutoff)] = {
            "R22_ORIGINAL": np.array_equal(original, fold["R22_ORIGINAL"]),
            "R22_PCT_PRESENT": np.array_equal(present_policy, fold["R22_PCT_PRESENT"]),
            "missing_exact_champion": np.array_equal(present_policy[~present], baseline[~present]),
        }
        if not all(checks["folds"][str(cutoff)].values()):
            raise BatchIntegrityError(f"Independent R22 fold policy recompute failed at {cutoff}")
    for view_name in REQUIRED_VIEWS:
        rows = np.asarray(views[view_name]["rows"], dtype=np.int64)
        if len(rows) != len(np.unique(rows)):
            raise BatchIntegrityError(f"R22 view {view_name} contains duplicate rows")
        checks["all_views"][view_name] = {
            "rows": int(len(rows)),
            "unique_rows": int(len(np.unique(rows))),
            "all_predictions_finite": all(np.isfinite(np.asarray(views[view_name][k])).all()
                                           for k in ("baseline", "R22_ORIGINAL", "R22_PCT_PRESENT")),
        }
    return checks


def _per_cycle_diagnostics(context: Mapping[str, Any],
                           folds: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    target = np.asarray(context["target"], dtype=np.float64)
    result: dict[str, Any] = {"gate_applied": False, "cycles": {}}
    for window, (cutoff, low, high) in WINDOWS.items():
        fold = folds[cutoff]
        for cycle in range(low, high + 1):
            mask = np.asarray(fold["cycle_codes"], dtype=np.int64) == cycle
            rows = np.asarray(fold["rows"], dtype=np.int64)[mask]
            actual = target[rows]
            raw = np.asarray(fold["raw_candidate"], dtype=np.float64)[mask]
            baseline = np.asarray(fold["baseline_prediction"], dtype=np.float64)[mask]
            pct = np.asarray(fold["cycle_percentile"], dtype=np.float64)[mask]
            rho_value = float(spearmanr(raw, actual).statistic) if len(rows) > 1 else float("nan")
            rho = rho_value if np.isfinite(rho_value) else None
            result["cycles"][str(cycle)] = {
                "origin": int(cutoff),
                "window": window,
                "n": int(len(rows)),
                "R22_ORIGINAL_mae": _mae(actual, np.asarray(fold["R22_ORIGINAL"])[mask]),
                "R22_PCT_PRESENT_mae": _mae(actual, np.asarray(fold["R22_PCT_PRESENT"])[mask]),
                "champion_mae": _mae(actual, baseline),
                "raw_prediction_spearman_vs_target": rho,
                "percentile_min": float(np.min(pct)),
                "percentile_max": float(np.max(pct)),
                "diagnostic_only": True,
            }
    return result


def _fit_history() -> list[dict[str, Any]]:
    if not LEDGER_PATH.is_file():
        return []
    history = []
    for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if (event.get("experiment_id") == "optimization_v3_final_continuation_R22"
                and event.get("event") in {"CONTINUATION_FIT_STARTED", "CONTINUATION_FIT_SAVED", "CONTINUATION_FIT_FAILED"}):
            history.append({"line_number": line_number, **event})
    return history


def _fit_receipts() -> dict[str, Any]:
    output = {}
    for cutoff in CONTINUATION_CUTOFFS:
        npz_path = RUN_DIR / f"R22_fit{cutoff}.npz"
        sidecar_path = npz_path.with_suffix(".json")
        if not npz_path.is_file() or not sidecar_path.is_file():
            raise BatchIntegrityError(f"R22 fit receipt is incomplete at cutoff {cutoff}")
        sidecar = _read_json(sidecar_path)
        metadata = sidecar["metadata"]
        output[str(cutoff)] = {
            "npz": str(npz_path.relative_to(ROOT)).replace("\\", "/"),
            "sidecar": str(sidecar_path.relative_to(ROOT)).replace("\\", "/"),
            "npz_sha256": file_sha256(npz_path),
            "sidecar_sha256": file_sha256(sidecar_path),
            "fit_seconds": float(metadata["fit_seconds"]),
            "fit_rows": int(metadata["fit_metadata"]["fit_rows"]),
            "score_rows": int(metadata["fit_metadata"]["score_rows"]),
            "fit_predictor_count": int(metadata["fit_predictor_count"]),
            "rss_after_fit_bytes": int(metadata["rss_after_fit_bytes"]),
            "resource_preflight": metadata["resource_preflight"],
        }
    return output


def build_result(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    folds = load_all_folds(context)
    views = _view_arrays(context, folds)
    independent = _independent_recompute(context, folds, views)
    policies = {name: _policy_metrics(context["frame"], context["target"], views, name)
                for name in ("R22_ORIGINAL", "R22_PCT_PRESENT")}
    qualifications = {name: _qualification(policy) for name, policy in policies.items()}
    original = policies["R22_ORIGINAL"]
    pct = policies["R22_PCT_PRESENT"]
    pct_selects = (
        qualifications["R22_PCT_PRESENT"]["qualified"]
        and pct["views"]["primary"]["candidate_mae"] <= original["views"]["primary"]["candidate_mae"]
        and (original["late_W5_W6"]["candidate_mae"] - pct["late_W5_W6"]["candidate_mae"]) >= 0.03
    )
    if pct_selects:
        selected = "R22_PCT_PRESENT"
        selection_reason = "PCT_PRESENT qualifies, primary MAE is no worse than ORIGINAL, and W5+W6 MAE improves by at least 0.03."
    elif qualifications["R22_ORIGINAL"]["qualified"]:
        selected = "R22_ORIGINAL"
        selection_reason = "PCT_PRESENT did not satisfy every predeclared selection condition; ORIGINAL qualifies."
    else:
        selected = "INCUMBENT_CHAMPION"
        selection_reason = "Neither R22 policy qualified; retain the incumbent champion."
    result = {
        "schema_version": 1,
        "experiment_id": "optimization_v3_final_continuation_R22",
        "status": "RESULT_VERIFIED",
        "freeze_signature": context["signature"],
        "source_head_before_fit": context.get("source_head"),
        "new_fits": 4,
        "reused_pilot_fits": [55, 67],
        "reused_pilot_bundle_hashes": context["pilot_hashes"],
        "primary_denominator": EXPECTED_PRIMARY_ROWS,
        "H12_denominator": EXPECTED_PRIMARY_ROWS,
        "W5_W6_denominator": EXPECTED_LATE_ROWS,
        "primary_H12_row_overlap": "same official Train row identities scored from different fit origins; not independent evidence",
        "policies": policies,
        "per_cycle_diagnostics": _per_cycle_diagnostics(context, folds),
        "qualification": qualifications,
        "selected_policy": selected,
        "selection_reason": selection_reason,
        "independent_verification": independent,
        "fit_receipts": _fit_receipts(),
        "fit_history": _fit_history(),
        "budget": {
            "before_joint_reservation": context["budget_snapshot_pre_reservation"],
            "after_execution": context["budget_snapshot"],
            "joint_reservation_key": JOINT_RESERVATION_KEY,
            "joint_reservation_card_id": JOINT_RESERVATION_CARD,
            "joint_reservation_fit_count": 8,
        },
        "official_test_used": False,
        "kaggle_access": False,
    }
    verification = {
        "status": "PASS",
        "freeze_signature": context["signature"],
        "independent_output_reconstruction": independent,
        "all_primary_window_rows_disjoint": len(set(np.concatenate([views[w]["rows"] for w in WINDOWS]).tolist())) == EXPECTED_PRIMARY_ROWS,
        "pilot_policy_reconstruction": {str(c): True for c in (55, 67)},
        "control_reconstruction": "exact all-six cache reconstruction validated before fit",
        "metric_formula": "mean(abs(threat_rank - prediction)); gain=champion MAE-candidate MAE",
        "gate_thresholds_unrounded": True,
        "source_data_boundary": {"official_train_only": True, "official_test_opened": False, "sample_submission_opened": False, "kaggle_calls": 0},
    }
    return result, verification


def publish_result(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result, verification = build_result(context)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in ((RESULT_PATH, result), (VERIFICATION_PATH, verification)):
        if path.exists():
            if _read_json(path) != payload:
                raise BatchIntegrityError(f"Existing R22 report differs; refusing rewrite: {path}")
        else:
            write_json_exclusive(path, payload)
    summary = _summary_markdown(result)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing R22 summary differs; refusing rewrite")
    else:
        with SUMMARY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary)
            stream.flush()
            os.fsync(stream.fileno())
    return result, verification


def _summary_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# R22 continuation result",
        "",
        f"- Status: `{result['status']}`",
        f"- Freeze signature: `{result['freeze_signature']}`",
        f"- Selected policy: **{result['selected_policy']}**",
        f"- Decision: {result['selection_reason']}",
        "- New predictor fits: 4 (cutoffs 59, 63, 71, 75); pilot 55/67 reused.",
        "- Evaluation labels are previously reused temporal Train development data; these are not untouched holdouts.",
        "- Primary and H12 have the same 38,588 Train row identities scored from different fit origins; they are not independent samples.",
        "",
        "## Metric summary",
        "",
        "MAE gains are champion MAE minus candidate MAE; positive values mean improvement. Gates use unrounded values.",
        "",
        "| Policy | View | n | Candidate MAE | Champion MAE | Gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ("R22_ORIGINAL", "R22_PCT_PRESENT"):
        metrics = result["policies"][policy]
        rows = [(name, item) for name, item in metrics["views"].items()]
        rows.append(("W5+W6", metrics["late_W5_W6"]))
        for view, item in rows:
            lines.append(f"| {policy} | {view} | {item['n']} | {item['candidate_mae']:.6f} | {item['champion_mae']:.6f} | {item['gain_vs_champion']:.6f} |")
    lines.extend(["", "## Qualification and group guards", ""])
    for policy in ("R22_ORIGINAL", "R22_PCT_PRESENT"):
        qualification = result["qualification"][policy]
        lines.append(f"### {policy}: {'QUALIFIED' if qualification['qualified'] else 'NOT QUALIFIED'}")
        for gate, passed in qualification["gates"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {gate}")
        lines.extend(["", "| View | Group | n | Harm (candidate − champion MAE) | Status |", "|---|---|---:|---:|---|"])
        for view, groups in result["policies"][policy]["groups"]["views"].items():
            for group, item in groups.items():
                status = ("PASS" if item["pass"] else "FAIL") if item["supported"] else "unsupported; descriptive"
                harm = "—" if item["harm"] is None else f"{item['harm']:.6f}"
                lines.append(f"| {view} | {group} | {item['n']} | {harm} | {status} |")
        lines.append("")
    lines.extend(["## Fit receipts", "", "| Cutoff | Fit rows | Score rows | Seconds | RSS after fit (bytes) | Bundle SHA-256 |", "|---:|---:|---:|---:|---:|---|"])
    for cutoff, receipt in result["fit_receipts"].items():
        lines.append(f"| {cutoff} | {receipt['fit_rows']} | {receipt['score_rows']} | {receipt['fit_seconds']:.3f} | {receipt['rss_after_fit_bytes']} | `{receipt['npz_sha256']}` |")
    lines.extend([
        "",
        "## Diagnostics and limits",
        "",
        "H12 early/late and per-cycle Spearman/MAE results are stored in `result.json` as descriptive diagnostics only; they do not change the predeclared gates.",
        "The percentile policy ranks raw predictions across every record in each complete cycle, including missing-stream records; only the output blend falls back to the exact champion for missing streams.",
        "No Test, SampleSubmission, or Kaggle data were accessed. No full-Train refit or submission is part of this continuation.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    context = prepare_context()
    if args.preflight_only:
        print(json.dumps(_preflight_payload(context), indent=2, allow_nan=False))
        return 0
    if args.freeze_only:
        print(json.dumps(freeze_context(context), indent=2, allow_nan=False))
        return 0
    result, verification = publish_result(context)
    print(json.dumps({"result": result, "verification": verification}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
