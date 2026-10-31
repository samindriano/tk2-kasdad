"""R18 runner for the frozen V3 batch1 queue card.

The default workflow remains explicit and staged: run ``--preflight-only`` for
read-only checks, ``--freeze-only`` to persist the reviewed recipe after MAIN's
source review, and ``--run-frozen`` only after the freeze exists. No model or
preprocessing estimators are fit by this calibration-only card.
"""
from __future__ import annotations

import json
import csv
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_reg_diagnostics import (
    FULL_CACHE, FULL_REPORT, PILOT_CACHE, PRIMARY_CACHE, _make_view,
)
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    category_novelty_flags,
    exclusive_training_slot,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    read_bundle,
    stream_missing_mask,
    write_bundle_exclusive,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r18 import (
    CARD_ID,
    MIN_HISTORY_ROWS,
    POLICY_BY_CUTOFF,
    R18IntegrityError,
    R18InsufficientHistory,
    apply_offset,
    estimate_recent_offset,
    fit_predict as r18_fit_predict,
    predict_with_policy,
)
from src.utils.paths import task_files


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R18.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R18"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R18"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R18"
PUBLICATION_PATH = REPORT_DIR / "publication.json"
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "990bb8484a6ee835ea1431bd3fb1738e957154e82bb47a689a0a6bda571f25b8"
DIAGNOSTICS_SHA256 = "8d634debd1dbc6a3eed5250593b9c213570b4fb139b0b9a623a8d704159916df"
OOF_CACHE_SHA256 = "283a6e91ebc904de6152103f2a1f20362d52bf82fd21172e2f8b37d48721e4a5"
DIAGNOSTICS_PATH = ROOT / "reports/optimization_v3/regression_diagnostics/diagnostics.json"
OOF_CACHE_PATH = ROOT / "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz"
SUBGROUP_REPORT = ROOT / "reports/optimization_v2/regression_repr/fullsix_subgroup_diagnostics.json"
RESIDUAL_FREEZE = ROOT / "experiments/optimization_v3/regression_residual/freeze.json"
DISTRIBUTIONAL_FREEZE = ROOT / "experiments/optimization_v3/regression_distributional/freeze.json"
RESIDUAL_FREEZE_SHA256 = "d8119aa2e3b745ffefe338d7407a468acd9d97075d7179bfe3f5abefdf873064"
DISTRIBUTIONAL_FREEZE_SHA256 = "cce68bf9b903f2a0f75691b4dabf3527374f6a06c0c285512472b961c244d39a"
HISTORY_CYCLES = frozenset({64, 65, 66, 67, 72, 73, 74, 75})
OOF_ARRAYS_WITHOUT_LABELS = {
    "primary_rows", "primary_cycle", "primary_prediction", "primary_lgb_svd32",
    "primary_hybrid_ridge", "primary_sparse_lgb_rank50", "h12_rows", "h12_cycle",
    "h12_prediction", "h12_lgb_svd32", "h12_hybrid_ridge", "h12_sparse_lgb_rank50",
}
EXPECTED_CARD = {
    "id": "R18",
    "task": "regression",
    "name": "Recent OOF median location offset",
    "family": "calibration_scalar",
    "priority": "P2",
    "change": "Latest four complete fit cycles of exact champion OOF: delta=clip(0.5*median(y-p),-2,2). Apply scalar future correction and clip0..100; cutoff55 unchanged. Uses median for MAE, superseding mean-offset draft.",
    "policy": "R_CALIBRATE",
    "pilot_predictor_fits": 0,
    "evaluation": "LATE",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "card_path": "reports/optimization_v3/research_backlog/cards/R18.md",
}
EXPECTED_PRIMARY_FOLDS = (
    (55, 56, 59, 4_114),
    (59, 60, 63, 6_493),
    (63, 64, 67, 7_078),
    (67, 68, 71, 4_950),
    (71, 72, 75, 7_462),
    (75, 76, 79, 8_491),
)
EXPECTED_LATE_VIEWS = {
    "W4": [(67, 68, 71)],
    "W6": [(75, 76, 79)],
    "primary": [(67, 68, 71), (75, 76, 79)],
    "H12_L": [(67, 68, 79)],
}
EXPECTED_DIAGNOSTIC_SOURCE_HASHES = {
    "data/raw/regression/Train.csv": TRAIN_SHA256,
    "experiments/runs/dual_reg_v1/stage1_paired_blends.npz": "1227404ffa7d9f5c636c109238bfc6ee99dcbfbc128c3a07f16a2eec7ddd0516",
    "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz": "eae6bea2dd6933de59ac0f63f54cc08124b59d4bcba91bbe05a7c5ec3dab9ff6",
    "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz": "4b87b0da34069ea3d75490233c574fbd92283f8b6472a43d8c654a20a4b904c9",
    "reports/optimization_v2/regression_repr/fullsix.json": "a9686473998722b8f86a0f6a4489b4e3a75a6b6ba20c661ac564e6a5e61aea48",
    "reports/optimization_v2/regression_repr/fullsix_subgroup_diagnostics.json": "6954fbdd9eb1c95794787ff54cf8a9b8c88dc97373ad9a8bf8e0aefdcaf1551e",
}
EXPECTED_REGRESSION_GATES = {
    "pilot_primary_MAE_gain_min": 0.03,
    "pilot_H12_MAE_harm_max": 0.05,
    "individual_pilot_view_MAE_harm_max": 0.5,
    "full_primary_MAE_gain_min": 0.03,
    "full_H12_MAE_harm_max": 0.05,
    "full_nonnegative_windows_min": 4,
    "supported_group_n_min": 100,
    "supported_tail_or_missing_MAE_harm_max": 0.25,
}
EXPECTED_SUPPORT_GROUPS = (
    "entry_friction0/1, stream present/missing, target tails for Regression, per-class and "
    "fit-defined text_load quartiles for Classification; unknown category status descriptive with denominators"
)


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _load_oof_predictions_only(path: Path) -> dict[str, np.ndarray]:
    """Load row/cycle/prediction arrays without opening either OOF target array."""
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen OOF cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        absent = sorted(OOF_ARRAYS_WITHOUT_LABELS - set(archive.files))
        if absent:
            raise BatchIntegrityError(f"Champion OOF cache lacks required prediction arrays: {absent}")
        return {key: archive[key] for key in OOF_ARRAYS_WITHOUT_LABELS}


def _array_sha256(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype == object:
        raise BatchIntegrityError("Object arrays cannot be part of a frozen R18 identity")
    payload = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(array.dtype.str.encode("ascii") + b"\0" + repr(array.shape).encode("ascii") + b"\0" + payload).hexdigest()


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r18.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_reg_diagnostics.py",
        ROOT / "src/utils/paths.py",
    ]


def _input_files(train_path: Path) -> dict[str, Path]:
    return {
        "official_regression_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_card": CARD_PATH,
        "accepted_regression_manifest": ROOT / "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json",
        "cache_only_diagnostics": DIAGNOSTICS_PATH,
        "current_champion_oof_cache": OOF_CACHE_PATH,
        "paired_lgb_ridge_cache": PRIMARY_CACHE,
        "sparse_regression_pilot_cache": PILOT_CACHE,
        "sparse_regression_fullsix_cache": FULL_CACHE,
        "regression_fullsix_report": FULL_REPORT,
        "regression_subgroup_diagnostics": SUBGROUP_REPORT,
        "regression_residual_freeze": RESIDUAL_FREEZE,
        "regression_distributional_freeze": DISTRIBUTIONAL_FREEZE,
    }


def _validate_card_and_queue(card: Mapping[str, Any], queue: Mapping[str, Any]) -> None:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256:
        raise BatchIntegrityError("Canonical queue bytes differ from the R18 frozen queue hash")
    if file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("Canonical R18 card bytes differ from its frozen card hash")
    if not isinstance(queue, dict) or load_queue(QUEUE_PATH) != dict(queue):
        raise BatchIntegrityError("Dispatcher queue object differs from the canonical queue file")
    cards = {str(item["id"]): item for item in queue["cards"]}
    if cards.get(CARD_ID) != dict(card):
        raise BatchIntegrityError("Dispatcher R18 card differs from the canonical queue entry")
    for name, expected in EXPECTED_CARD.items():
        if card.get(name) != expected:
            raise BatchIntegrityError(f"R18 canonical card field changed: {name}")
    if card.get("pilot_predictor_fits") != 0:
        raise BatchIntegrityError("R18 frozen predictor-fit count must remain exactly zero")
    if queue["baseline"]["regression"].get("Train_sha256") != TRAIN_SHA256:
        raise BatchIntegrityError("Canonical Regression Train hash changed")
    if queue["baseline"]["regression"].get("manifest_sha256") != "dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a":
        raise BatchIntegrityError("Accepted Regression manifest hash changed")
    regression_gate = queue["evaluation"]["regression_gate"]
    if dict(regression_gate) != EXPECTED_REGRESSION_GATES:
        raise BatchIntegrityError("R18 frozen Regression pilot/full/subgroup gates changed")
    if queue["evaluation"].get("late_gate_note") != (
        "Apply same aggregate delta thresholds to explicitly labelled W4+W6 and H12_L; report early fallback separately. "
        "No external promotion from late-only pilot. No downgrading gates after scores."
    ):
        raise BatchIntegrityError("R18 LATE gate scope/fallback note changed")
    if queue["evaluation"].get("support_groups") != EXPECTED_SUPPORT_GROUPS:
        raise BatchIntegrityError("R18 frozen support-group evaluation contract changed")


def _load_official_train() -> tuple[Path, pd.DataFrame, np.ndarray, dict[int, float]]:
    # Resolve and open only the official Train. Test and SampleSubmission are
    # neither opened nor queried by this bridge.
    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from the frozen R18 input")
    # Exclude the target column from the full-table read. Only labels from the
    # two declared past OOF history blocks are extracted below; validation and
    # later-cycle targets never enter this preflight frame.
    header = pd.read_csv(train_path, nrows=0).columns.tolist()
    if "threat_rank" not in header or "cycle_code" not in header:
        raise BatchIntegrityError("Official Regression Train lacks target/cycle columns")
    frame = pd.read_csv(train_path, usecols=lambda name: name != "threat_rank")
    required = {
        "record_id", "cycle_code", "impact_load", "scope_pressure", "text_load",
        "scope_trace", "entry_friction", "org_code", "module_code", "pattern_code",
        "signal_stream",
    }
    absent = sorted(required - set(frame.columns))
    if absent or len(frame) != 50_608:
        raise BatchIntegrityError(f"Official Regression Train schema/size mismatch: missing={absent}, rows={len(frame)}")
    if frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise BatchIntegrityError("Official Regression Train has blank record_id values")
    if frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Regression Train record_id values are not unique")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Regression Train cycle coverage differs from 47–79")
    cycle_index = header.index("cycle_code")
    target_index = header.index("threat_rank")
    history_targets: dict[int, float] = {}
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row_position, record in enumerate(reader):
            if row_position >= len(frame):
                raise BatchIntegrityError("CSV row count changed while extracting past-only targets")
            cycle = int(record[cycle_index])
            if cycle in HISTORY_CYCLES:
                value = float(record[target_index])
                if not np.isfinite(value) or not 0 <= value <= 100:
                    raise BatchIntegrityError(f"Invalid past OOF target at row {row_position}")
                history_targets[row_position] = value
    if len(history_targets) != int(np.isin(cycles, list(HISTORY_CYCLES)).sum()):
        raise BatchIntegrityError("Past-only target extraction did not cover the declared history rows")
    # _make_view expects a target column but only uses these target values to
    # populate diagnostics. Mask every non-history outcome before its call.
    safe_target = np.full(len(frame), np.nan, dtype=np.float64)
    for row, value in history_targets.items():
        safe_target[int(row)] = value
    frame["threat_rank"] = safe_target
    return train_path, frame, cycles, history_targets


def _validate_evidence_hashes(
    queue: Mapping[str, Any], input_files: Mapping[str, Path],
) -> tuple[dict[str, str], dict[str, Any]]:
    baseline = queue["baseline"]["regression"]
    manifest_path = input_files["accepted_regression_manifest"]
    if file_sha256(manifest_path) != baseline["manifest_sha256"]:
        raise BatchIntegrityError("Accepted Regression manifest differs from the canonical queue hash")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "regression" or manifest.get("status") != "PREPARED_AND_QA_PASS":
        raise BatchIntegrityError("Accepted Regression manifest is not the expected QA-passed control")
    diagnostics_hash = file_sha256(DIAGNOSTICS_PATH)
    if diagnostics_hash != DIAGNOSTICS_SHA256:
        raise BatchIntegrityError("Regression diagnostics file differs from the reviewed row/cutoff evidence")
    if file_sha256(OOF_CACHE_PATH) != OOF_CACHE_SHA256:
        raise BatchIntegrityError("Current-champion OOF cache differs from the reviewed cache hash")
    if file_sha256(RESIDUAL_FREEZE) != RESIDUAL_FREEZE_SHA256:
        raise BatchIntegrityError("Regression residual-head freeze differs from the reviewed lineage hash")
    if file_sha256(DISTRIBUTIONAL_FREEZE) != DISTRIBUTIONAL_FREEZE_SHA256:
        raise BatchIntegrityError("Regression distributional freeze differs from the reviewed lineage hash")
    residual_freeze = json.loads(RESIDUAL_FREEZE.read_text(encoding="utf-8"))
    residual_inputs = residual_freeze.get("source_and_input_sha256", {})
    if (residual_freeze.get("test_read") is not False
            or residual_freeze.get("official_train_sha256") != TRAIN_SHA256
            or residual_inputs.get("experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz") != OOF_CACHE_SHA256):
        raise BatchIntegrityError("Residual-head frozen provenance does not bind the exact OOF cache/Train with test_read=false")
    distributional_freeze = json.loads(DISTRIBUTIONAL_FREEZE.read_text(encoding="utf-8"))
    distributional_inputs = {
        str(path).replace("\\", "/"): digest
        for path, digest in distributional_freeze.get("identity", {}).get("inputs", {}).items()
    }
    if (distributional_inputs.get("experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz") != OOF_CACHE_SHA256
            or distributional_inputs.get("reports/optimization_v3/regression_diagnostics/diagnostics.json") != DIAGNOSTICS_SHA256
            or distributional_inputs.get("data/raw/regression/Train.csv") != TRAIN_SHA256):
        raise BatchIntegrityError("Distributional frozen provenance does not bind the exact OOF/diagnostics/Train inputs")
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    if diagnostics.get("ignored_row_aligned_cache") != "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz":
        raise BatchIntegrityError("Diagnostics do not name the expected row-aligned OOF cache")
    historical_hashes = diagnostics.get("source_and_cache_sha256", {})
    observed_sources: dict[str, str] = {}
    for relative, expected in EXPECTED_DIAGNOSTIC_SOURCE_HASHES.items():
        path = ROOT / relative
        observed = file_sha256(path)
        if historical_hashes.get(relative) != expected or observed != expected:
            raise BatchIntegrityError(f"Frozen diagnostics source/cache hash mismatch: {relative}")
        observed_sources[relative] = observed
    observed_inputs = {name: file_sha256(path) for name, path in sorted(input_files.items())}
    return observed_inputs, {
        "observed_source_hashes": observed_sources, "diagnostics": diagnostics,
        "residual_freeze": residual_freeze,
        "distributional_freeze": distributional_freeze,
    }


def _position_index(rows: np.ndarray) -> dict[int, int]:
    if rows.ndim != 1 or len(rows) != len(np.unique(rows)):
        raise BatchIntegrityError("OOF row positions must be a unique vector")
    return {int(row): index for index, row in enumerate(rows)}


def _assert_exact_prediction(label: str, observed: np.ndarray, expected: np.ndarray) -> None:
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        maximum = float(np.max(np.abs(observed - expected))) if observed.shape == expected.shape and len(observed) else None
        raise BatchIntegrityError(f"{label} is not an exact champion reconstruction; max_abs={maximum}")


def _validate_primary_oof_mapping(
    *, frame: pd.DataFrame, cycles: np.ndarray, history_targets: Mapping[int, float],
    diagnostics: Mapping[str, Any], residual_freeze: Mapping[str, Any],
    oof: Mapping[str, np.ndarray],
    sparse_cache: Mapping[str, np.ndarray], base_cache: Mapping[str, np.ndarray],
) -> tuple[dict[int, int], dict[str, Any]]:
    required_oof = {
        "primary_rows", "primary_cycle", "primary_prediction",
        "primary_lgb_svd32", "primary_hybrid_ridge", "primary_sparse_lgb_rank50",
        "h12_rows", "h12_cycle", "h12_prediction",
        "h12_lgb_svd32", "h12_hybrid_ridge", "h12_sparse_lgb_rank50",
    }
    if not required_oof.issubset(oof):
        raise BatchIntegrityError(f"Champion OOF cache is missing fields: {sorted(required_oof - set(oof))}")
    rows = np.asarray(oof["primary_rows"], dtype=np.int64)
    oof_cycles = np.asarray(oof["primary_cycle"], dtype=np.int64)
    if len(rows) == 0 or len(rows) != len(np.unique(rows)) or np.any((rows < 0) | (rows >= len(frame))):
        raise BatchIntegrityError("Champion OOF rows are empty, duplicated, or out of Train bounds")
    if not np.array_equal(oof_cycles, cycles[rows]):
        raise BatchIntegrityError("Champion OOF cycle values do not align to official Train row positions")
    expected_oof_rows = np.flatnonzero((cycles >= 56) & (cycles <= 79))
    if not np.array_equal(np.sort(rows), expected_oof_rows):
        raise BatchIntegrityError("Champion OOF primary row population differs from cycles 56–79")
    position = _position_index(rows)

    audits = diagnostics.get("fit_and_row_audits", {}).get("primary", [])
    if len(audits) != len(EXPECTED_PRIMARY_FOLDS):
        raise BatchIntegrityError("Diagnostics primary fold audit count differs from the six expected source cutoffs")
    source_cutoff_by_cycle: dict[int, int] = {}
    fold_reconstruction = []
    for (cutoff, low, high, expected_n), audit in zip(EXPECTED_PRIMARY_FOLDS, audits):
        expected_view = f"{low}-{high}"
        if (int(audit.get("cutoff", -1)) != cutoff or audit.get("view") != expected_view
                or int(audit.get("validation_rows", -1)) != expected_n
                or float(audit.get("base_ridge25_reconstruction_max_abs_error", float("inf"))) != 0.0):
            raise BatchIntegrityError(f"Diagnostics source-cutoff mapping changed for {expected_view}")
        for cycle in range(low, high + 1):
            if cycle in source_cutoff_by_cycle:
                raise BatchIntegrityError(f"Diagnostics map cycle {cycle} to multiple fit cutoffs")
            source_cutoff_by_cycle[cycle] = cutoff
        block, _ = _make_view(
            frame, cutoff, low, high, sparse_cache, base_cache,
            view_name=f"R18 OOF source {low}-{high}",
        )
        block_rows = block["row"].to_numpy(dtype=np.int64)
        block_positions = np.asarray([position[int(row)] for row in block_rows], dtype=np.int64)
        expected_rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        if not np.array_equal(block_rows, expected_rows) or len(block_rows) != expected_n:
            raise BatchIntegrityError(f"Exact champion fold rows differ from diagnostics at {cutoff}")
        _assert_exact_prediction(
            f"primary OOF champion cutoff {cutoff}",
            np.asarray(oof["primary_prediction"], dtype=np.float64)[block_positions],
            block["prediction"].to_numpy(dtype=np.float64),
        )
        for source_key, view_key in (
            ("primary_lgb_svd32", "lgb_svd32"),
            ("primary_hybrid_ridge", "hybrid_ridge"),
            ("primary_sparse_lgb_rank50", "sparse_lgb_rank50"),
        ):
            _assert_exact_prediction(
                f"{source_key} cutoff {cutoff}",
                np.asarray(oof[source_key], dtype=np.float64)[block_positions],
                block[view_key].to_numpy(dtype=np.float64),
            )
        fold_reconstruction.append({
            "source_cutoff": cutoff, "scored_cycles": [low, high],
            "rows": int(len(block_rows)), "component_and_champion_exact": True,
        })

    # The history labels are checked only for the two past-only calibration
    # windows. No R18 offset or candidate score metric is computed in preflight.
    history_audits = []
    for history_cycles, source_cutoff, future_start in (
        ((64, 65, 66, 67), 63, 68),
        ((72, 73, 74, 75), 71, 76),
    ):
        selected_positions = np.flatnonzero(np.isin(oof_cycles, history_cycles))
        selected_rows = rows[selected_positions]
        selected_cycles = oof_cycles[selected_positions]
        if len(selected_rows) < MIN_HISTORY_ROWS:
            raise BatchIntegrityError(f"R18 history support is below {MIN_HISTORY_ROWS}: {len(selected_rows)}")
        if len(np.unique(selected_rows)) != len(selected_rows):
            raise BatchIntegrityError("R18 history rows overlap or duplicate")
        if tuple(sorted(np.unique(selected_cycles).tolist())) != history_cycles:
            raise BatchIntegrityError(f"R18 history does not include all four complete cycles: {history_cycles}")
        mapped_cutoffs = np.asarray([source_cutoff_by_cycle.get(int(c), -1) for c in selected_cycles], dtype=np.int64)
        if not np.array_equal(np.unique(mapped_cutoffs), np.asarray([source_cutoff], dtype=np.int64)):
            raise BatchIntegrityError(f"R18 history {history_cycles} does not map wholly to cutoff {source_cutoff}")
        if np.any(mapped_cutoffs >= selected_cycles) or np.any(mapped_cutoffs >= future_start):
            raise BatchIntegrityError("R18 OOF history includes same/future-cycle labels")
        observed_target = np.asarray([history_targets[int(row)] for row in selected_rows], dtype=np.float64)
        if not np.isfinite(observed_target).all():
            raise BatchIntegrityError("R18 past-only Train targets are missing or nonfinite")
        freeze_folds = residual_freeze.get("actual_oof_fold_manifests", [])
        matching_freeze_fold = next((fold for fold in freeze_folds
            if int(fold.get("outer_fit_cutoff", -1)) == source_cutoff
            and fold.get("cycle_range") == [history_cycles[0], history_cycles[-1]]), None)
        freeze_hash_check = "not_in_residual_fold_manifest; diagnostics and exact-cache mapping checked"
        if matching_freeze_fold is not None:
            row_hash = hashlib.sha256(np.ascontiguousarray(selected_rows, dtype=np.int64).tobytes()).hexdigest()
            target_hash = hashlib.sha256(np.ascontiguousarray(observed_target, dtype=np.float64).tobytes()).hexdigest()
            if (matching_freeze_fold.get("rows") != len(selected_rows)
                    or matching_freeze_fold.get("row_positions_sha256") != row_hash
                    or matching_freeze_fold.get("target_sha256") != target_hash):
                raise BatchIntegrityError(f"Residual freeze history lineage hash mismatch for cutoff {source_cutoff}")
            freeze_hash_check = "PASS"
        history_audits.append({
            "history_cycles": list(history_cycles), "source_cutoff": source_cutoff,
            "first_future_cycle": future_start, "rows": int(len(selected_rows)),
            "row_positions_sha256": _array_sha256(selected_rows),
            "exact_champion_prediction_sha256": _array_sha256(
                np.asarray(oof["primary_prediction"], dtype=np.float64)[selected_positions]
            ),
            "complete_cycles": True, "unique_rows": True,
            "past_only_targets_loaded": True,
            "residual_freeze_fold_hash_check": freeze_hash_check,
            "strictly_past_only": True,
        })
    return source_cutoff_by_cycle, {
        "primary_oof_rows": int(len(rows)), "primary_oof_rows_unique": True,
        "primary_oof_cycle_alignment": True, "fold_reconstruction": fold_reconstruction,
        "calibration_history_support": history_audits,
    }


def _reconstruct_control(
    *, frame: pd.DataFrame, cycles: np.ndarray, cutoff: int, low: int, high: int,
    sparse_cache: Mapping[str, np.ndarray], base_cache: Mapping[str, np.ndarray],
    oof: Mapping[str, np.ndarray], include_arrays: bool = False,
) -> dict[str, Any]:
    block, audit = _make_view(
        frame, cutoff, low, high, sparse_cache, base_cache,
        view_name=f"R18 exact champion score {low}-{high}",
    )
    rows = block["row"].to_numpy(dtype=np.int64)
    expected = np.flatnonzero((cycles >= low) & (cycles <= high))
    if not np.array_equal(rows, expected) or len(np.unique(rows)) != len(rows):
        raise BatchIntegrityError(f"R18 control rows are not exact for cutoff {cutoff}: {low}-{high}")
    prediction = block["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 100.0)):
        raise BatchIntegrityError(f"R18 exact champion control is invalid at cutoff {cutoff}")
    if cutoff in (55, 67):
        hrows = np.asarray(oof["h12_rows"], dtype=np.int64)
        hcycles = np.asarray(oof["h12_cycle"], dtype=np.int64)
        if (len(hrows) != len(np.unique(hrows)) or not np.array_equal(hcycles, cycles[hrows])
                or not np.array_equal(np.sort(hrows), np.flatnonzero((cycles >= 56) & (cycles <= 79)))):
            raise BatchIntegrityError("H12 OOF rows/cycles do not map uniquely to official Train")
        hpositions = _position_index(hrows)
        diagnostic_positions = np.asarray([hpositions[int(row)] for row in rows], dtype=np.int64)
        _assert_exact_prediction(
            f"H12 diagnostic champion cutoff {cutoff}",
            np.asarray(oof["h12_prediction"], dtype=np.float64)[diagnostic_positions],
            prediction,
        )
        for key, view_key in (
            ("h12_lgb_svd32", "lgb_svd32"),
            ("h12_hybrid_ridge", "hybrid_ridge"),
            ("h12_sparse_lgb_rank50", "sparse_lgb_rank50"),
        ):
            _assert_exact_prediction(
                f"{key} cutoff {cutoff}",
                np.asarray(oof[key], dtype=np.float64)[diagnostic_positions],
                block[view_key].to_numpy(dtype=np.float64),
            )
    else:
        prows = np.asarray(oof["primary_rows"], dtype=np.int64)
        ppositions = _position_index(prows)
        diagnostic_positions = np.asarray([ppositions[int(row)] for row in rows], dtype=np.int64)
        _assert_exact_prediction(
            f"primary OOF champion cutoff {cutoff}",
            np.asarray(oof["primary_prediction"], dtype=np.float64)[diagnostic_positions],
            prediction,
        )
    output = {
        "cutoff": cutoff, "cycles": [low, high], "rows": int(len(rows)),
        "row_exact": True, "champion_reconstructed": True,
        "diagnostic_component_match": True, "ridge25_reconstruction_max_abs_error": audit["base_ridge25_reconstruction_max_abs_error"],
        "row_positions_sha256": _array_sha256(rows),
        "champion_prediction_sha256": _array_sha256(prediction),
        "component_predictions_sha256": _array_sha256(np.column_stack((
            block["lgb_svd32"].to_numpy(dtype=np.float64),
            block["hybrid_ridge"].to_numpy(dtype=np.float64),
            block["sparse_lgb_rank50"].to_numpy(dtype=np.float64),
        ))),
    }
    if include_arrays:
        output["_arrays"] = {
            "cutoff": int(cutoff),
            "rows": rows,
            "cycle": cycles[rows],
            "baseline_prediction": prediction,
            "record_id": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
        }
    return output


def _synthetic_smoke_checks() -> list[str]:
    cycles = np.repeat(np.arange(64, 68, dtype=np.int64), 125)
    rows = np.arange(500, dtype=np.int64)
    source = np.full(500, 63, dtype=np.int64)
    pred = np.full(500, 50.0, dtype=np.float64)
    score_cycles = np.full(9, 68, dtype=np.int64)
    baseline = np.full(9, 50.0, dtype=np.float64)
    common = {
        "history_rows": rows, "history_cycle": cycles,
        "history_source_cutoff": source, "expected_history_cycles": (64, 65, 66, 67),
        "expected_source_cutoff": 63, "score_cycle": score_cycles,
    }
    passed: list[str] = []
    delta, _ = estimate_recent_offset(**common, history_target=pred + 1.0, history_prediction=pred)
    assert delta == 0.5 and np.all(apply_offset(baseline, delta) > baseline)
    passed.append("positive_underprediction_sign")
    delta, _ = estimate_recent_offset(**common, history_target=pred - 1.0, history_prediction=pred)
    assert delta == -0.5 and np.all(apply_offset(baseline, delta) < baseline)
    passed.append("negative_overprediction_sign")
    for residual, expected in ((20.0, 2.0), (-20.0, -2.0)):
        delta, _ = estimate_recent_offset(**common, history_target=pred + residual, history_prediction=pred)
        assert delta == expected
    passed.append("offset_cap_both_directions")
    assert np.array_equal(apply_offset(np.array([0.0, 100.0]), 2.0), np.array([2.0, 100.0]))
    assert np.array_equal(apply_offset(np.array([0.0, 100.0]), -2.0), np.array([0.0, 98.0]))
    passed.append("prediction_clip_at_target_bounds")
    early_baseline = np.full(12, 50.0, dtype=np.float64)
    early_cycles = np.arange(56, 68, dtype=np.int64)
    unchanged, state = predict_with_policy(cutoff=55, baseline_prediction=early_baseline, score_cycle=early_cycles, history=None)
    assert np.array_equal(unchanged, early_baseline) and state["status"] == "EXACT_CHAMPION_FALLBACK"
    late_baseline = np.full(12, 50.0, dtype=np.float64)
    late_cycles = np.arange(68, 80, dtype=np.int64)
    unchanged, state = predict_with_policy(cutoff=67, baseline_prediction=late_baseline, score_cycle=late_cycles, history=None)
    assert np.array_equal(unchanged, late_baseline) and state["reason"] == "OOF_history_unavailable"
    passed.append("cutoff55_and_early_fallback")
    too_small = {**common, "history_rows": rows[:-1], "history_cycle": cycles[:-1],
                 "history_source_cutoff": source[:-1]}
    try:
        estimate_recent_offset(**too_small, history_target=pred[:-1] + 1, history_prediction=pred[:-1])
    except R18InsufficientHistory:
        pass
    else:
        raise AssertionError("499-row history did not trigger the frozen support gate")
    passed.append("minimum_support_gate")
    duplicate_rows = rows.copy(); duplicate_rows[-1] = duplicate_rows[-2]
    try:
        estimate_recent_offset(**{**common, "history_rows": duplicate_rows},
                               history_target=pred + 1, history_prediction=pred)
    except R18IntegrityError:
        pass
    else:
        raise AssertionError("duplicate OOF history rows were not rejected")
    passed.append("history_overlap_rejected")
    try:
        estimate_recent_offset(**{**common, "history_source_cutoff": np.full(500, 64)},
                               history_target=pred + 1, history_prediction=pred)
    except R18IntegrityError:
        pass
    else:
        raise AssertionError("same-cycle OOF source cutoff was not rejected")
    passed.append("same_or_future_source_cutoff_rejected")
    try:
        estimate_recent_offset(**{**common, "score_cycle": np.full(9, 67)},
                               history_target=pred + 1, history_prediction=pred)
    except R18IntegrityError:
        pass
    else:
        raise AssertionError("history/score cycle overlap was not rejected")
    passed.append("history_score_overlap_rejected")
    adapter_result = r18_fit_predict(
        fit_x=None, fit_y=None, score_x=None,
        controls={
            "baseline_prediction": np.full(12, 50.0),
            "score_cycle": np.arange(68, 80, dtype=np.int64),
            "history_rows": rows,
            "history_cycle": cycles,
            "history_source_cutoff": source,
            "history_target": pred + 1.0,
            "history_prediction": pred,
        },
        card_id=CARD_ID, cutoff=67,
    )
    assert adapter_result["fit_predictor_count"] == 0
    assert adapter_result["state"]["delta"] == 0.5
    assert np.all(adapter_result["prediction"] > 50.0)
    passed.append("adapter_zero_fit_and_policy_wiring")

    synthetic_preflight = {
        "input_sha256": {"train": "a" * 64},
        "source_sha256": {"runner": "b" * 64},
        "canonical_views": EXPECTED_LATE_VIEWS,
        "regression_gate_contract": EXPECTED_REGRESSION_GATES,
        "support_group_contract": EXPECTED_SUPPORT_GROUPS,
        "fit_resource_budget": {
            "predictor_fits": 0, "preprocessing_fits": 0,
            "new_model_objects": 0, "official_fitting_started": False,
        },
        "exact_champion_control_identity": {"67": {"rows": 12}},
        "oof_row_cutoff_mapping": {"calibration_history_support": []},
        "accepted_manifest_sha256": "c" * 64,
        "frozen_oof_lineage": {"test_read": False},
    }
    synthetic_identity = _freeze_identity({"id": CARD_ID}, synthetic_preflight)
    synthetic_document = _freeze_document(synthetic_identity)
    assert _verify_freeze_document(synthetic_document, synthetic_identity) == synthetic_document["signature"]
    assert synthetic_identity["future_target_values_in_freeze"] is False
    assert synthetic_identity["candidate_delta_in_freeze"] is False
    assert "delta" not in synthetic_identity and "candidate_predictions" not in synthetic_identity
    tampered = dict(synthetic_document)
    tampered["identity"] = {**synthetic_identity, "recipe": {"changed": True}}
    try:
        _verify_freeze_document(tampered, synthetic_identity)
    except BatchIntegrityError:
        pass
    else:
        raise AssertionError("tampered freeze identity/signature was accepted")
    passed.append("freeze_identity_signature_and_no_outcome_payload")

    synthetic_cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 25)
    synthetic_rows = np.arange(len(synthetic_cycles), dtype=np.int64)
    synthetic_frame = pd.DataFrame({
        "record_id": [f"syn-{i}" for i in synthetic_rows],
        "cycle_code": synthetic_cycles,
        "entry_friction": np.tile(np.arange(25) % 2, 33),
        "signal_stream": np.where(np.tile(np.arange(25) % 2, 33) == 0, "", "x"),
        "org_code": np.tile(["o1", "o2", "o1", "o3", "o2"], 165),
        "module_code": np.tile(["m1", "m2", "m3", "m1", "m2"], 165),
        "pattern_code": np.tile(["p1", "p2", "p1", "p2", "p3"], 165),
    })
    synthetic_target = {int(row): 51.0 for row in synthetic_rows if 56 <= synthetic_cycles[row] <= 79}
    synthetic_blocks: dict[int, dict[str, Any]] = {}
    for cutoff, low, high, corrected in ((55, 56, 67, False), (67, 68, 79, True), (75, 76, 79, True)):
        selected_rows = np.flatnonzero((synthetic_cycles >= low) & (synthetic_cycles <= high))
        baseline_values = np.full(len(selected_rows), 50.0)
        candidate_values = baseline_values + 1.0 if corrected else baseline_values.copy()
        synthetic_blocks[cutoff] = {
            "cutoff": cutoff, "rows": selected_rows,
            "baseline_prediction": baseline_values, "prediction": candidate_values,
            "record_id": synthetic_frame.iloc[selected_rows]["record_id"].astype(str).to_numpy(dtype=str),
        }
    synthetic_result = _evaluate_frozen_predictions(
        blocks=synthetic_blocks, cycles=synthetic_cycles,
        frame=synthetic_frame, target_by_row=synthetic_target,
    )
    assert synthetic_result["views"]["primary"]["rows"] == 200
    assert synthetic_result["views"]["H12_E"]["candidate_mae"] == synthetic_result["views"]["H12_E"]["baseline_mae"]
    assert synthetic_result["all_pilot_gates_pass"] is True
    passed.append("synthetic_late_views_and_subgroup_gate_aggregation")
    w4_block = synthetic_blocks[67]
    w4_rows = np.asarray(w4_block["rows"], dtype=np.int64)
    w4_mask = (synthetic_cycles[w4_rows] >= 68) & (synthetic_cycles[w4_rows] <= 71)
    w4_rows = w4_rows[w4_mask]
    w4_view = {
        "rows": w4_rows, "cycle": synthetic_cycles[w4_rows],
        "cutoff": np.full(len(w4_rows), 67, dtype=np.int64),
        "baseline": np.asarray(w4_block["baseline_prediction"])[w4_mask],
        "candidate": np.asarray(w4_block["prediction"])[w4_mask],
    }
    w4_target = np.asarray([synthetic_target[int(row)] for row in w4_rows], dtype=np.float64)
    w4_subgroups = synthetic_result["subgroups"]["W4"]
    check = _verify_subgroup_table(
        view_name="W4", view=w4_view, frame=synthetic_frame, target=w4_target,
        stored=w4_subgroups,
    )
    assert check["status"] == "PASS" and check["groups"] > 0
    tampered_subgroups = json.loads(json.dumps(w4_subgroups))
    tampered_subgroups["target_tail"]["middle"]["n"] += 1
    try:
        _verify_subgroup_table(
            view_name="W4", view=w4_view, frame=synthetic_frame, target=w4_target,
            stored=tampered_subgroups,
        )
    except BatchIntegrityError:
        pass
    else:
        raise AssertionError("tampered R18 subgroup denominator was accepted")
    passed.append("independent_subgroup_table_denominators_and_metrics")
    middle_rows = np.arange(100, 200, dtype=np.int64)
    middle_view = {
        "rows": middle_rows,
        "cycle": synthetic_cycles[middle_rows],
        "cutoff": np.full(len(middle_rows), 55, dtype=np.int64),
        "baseline": np.full(len(middle_rows), 50.0),
        "candidate": np.full(len(middle_rows), 51.0),
    }
    middle_targets = {int(row): 50.0 for row in middle_rows}
    middle_summary, middle_failures = _subgroup_table(
        view_name="W4", view=middle_view, frame=synthetic_frame,
        target_by_row=middle_targets, support_min=100,
    )
    assert middle_summary["target_tail"]["middle"]["candidate_minus_baseline_mae"] == 1.0
    assert middle_summary["target_tail"]["middle"]["guard_applies"] is False
    assert middle_failures == []
    passed.append("middle_only_tail_guard_harm_is_descriptive")
    sample_rows = np.arange(100, dtype=np.int64)
    sample_view = {
        "rows": sample_rows, "cycle": synthetic_cycles[sample_rows],
        "cutoff": np.full(len(sample_rows), 55, dtype=np.int64),
    }
    sample_labels = _group_labels_for_view(
        view=sample_view, frame=synthetic_frame, target=np.full(len(sample_rows), 51.0),
    )
    assert set(sample_labels["signal_stream"]) == {"present", "missing"}
    passed.append("series_input_to_stream_missing_grouping")
    return passed


def _prepare_preflight(
    card: Mapping[str, Any], queue: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_card_and_queue(card, queue)
    train_path, frame, cycles, history_targets = _load_official_train()
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_LATE_VIEWS:
        raise BatchIntegrityError(f"Canonical R18 LATE evaluation views changed: {views}")
    inputs = _input_files(train_path)
    input_hashes, evidence = _validate_evidence_hashes(queue, inputs)
    diagnostics = evidence["diagnostics"]
    base_cache = _load_archive(PRIMARY_CACHE)
    pilot_cache = _load_archive(PILOT_CACHE)
    full_cache = _load_archive(FULL_CACHE)
    sparse_cache = {**pilot_cache, **full_cache}
    oof = _load_oof_predictions_only(OOF_CACHE_PATH)

    source_cutoff_by_cycle, mapping_audit = _validate_primary_oof_mapping(
        frame=frame, cycles=cycles, history_targets=history_targets, diagnostics=diagnostics,
        residual_freeze=evidence["residual_freeze"], oof=oof,
        sparse_cache=sparse_cache, base_cache=base_cache,
    )
    del source_cutoff_by_cycle
    control_details = [
        _reconstruct_control(
            frame=frame, cycles=cycles, cutoff=67, low=68, high=79,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof, include_arrays=True,
        ),
        _reconstruct_control(
            frame=frame, cycles=cycles, cutoff=75, low=76, high=79,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof, include_arrays=True,
        ),
        _reconstruct_control(
            frame=frame, cycles=cycles, cutoff=55, low=56, high=67,
            sparse_cache=sparse_cache, base_cache=base_cache, oof=oof, include_arrays=True,
        ),
    ]
    controls = [{k: v for k, v in item.items() if k != "_arrays"} for item in control_details]
    controls_by_cutoff = {int(item["cutoff"]): item["_arrays"] for item in control_details}
    # This is a policy-only check for the early fallback. It does not compute
    # an R18 delta or score candidate predictions against validation outcomes.
    early_rows = np.flatnonzero((cycles >= 56) & (cycles <= 67))
    early_baseline = controls_by_cutoff[55]["baseline_prediction"]
    unchanged, fallback_state = predict_with_policy(
        cutoff=55, baseline_prediction=early_baseline,
        score_cycle=cycles[early_rows], history=None,
    )
    if not np.array_equal(unchanged, early_baseline) or fallback_state.get("status") != "EXACT_CHAMPION_FALLBACK":
        raise BatchIntegrityError("R18 H12_E exact-champion fallback policy is not identity")

    if int(card["pilot_predictor_fits"]) != 0:
        raise BatchIntegrityError("R18 resource budget must reserve zero predictor fits")
    source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in _source_files()}
    ending_input_hashes = {name: file_sha256(path) for name, path in sorted(inputs.items())}
    ending_source_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in _source_files()}
    if ending_input_hashes != input_hashes or ending_source_hashes != source_hashes:
        raise BatchIntegrityError("A frozen R18 input/source changed during preflight; retry from a clean review state")
    smoke = _synthetic_smoke_checks()
    preflight = {
        "card_id": CARD_ID,
        "status": "READ_ONLY_PREFLIGHT_PASS",
        "scope": "preflight only; no official R18 offset, candidate prediction, metric, freeze, fit, or output write",
        "queue_sha256": file_sha256(QUEUE_PATH),
        "card_sha256": file_sha256(CARD_PATH),
        "train_sha256": file_sha256(train_path),
        "accepted_manifest_sha256": input_hashes["accepted_regression_manifest"],
        "diagnostics_sha256": file_sha256(DIAGNOSTICS_PATH),
        "oof_cache_sha256": file_sha256(OOF_CACHE_PATH),
        "frozen_oof_lineage": {
            "regression_residual_freeze_sha256": file_sha256(RESIDUAL_FREEZE),
            "regression_distributional_freeze_sha256": file_sha256(DISTRIBUTIONAL_FREEZE),
            "residual_freeze_test_read": evidence["residual_freeze"]["test_read"],
            "residual_freeze_binds_oof_cache": True,
            "distributional_freeze_binds_oof_cache_and_diagnostics": True,
        },
        "input_sha256": input_hashes,
        "other_input_sha256": {k: v for k, v in input_hashes.items() if k not in {
            "canonical_queue", "canonical_card", "official_regression_train",
            "accepted_regression_manifest", "cache_only_diagnostics", "current_champion_oof_cache",
        }},
        "source_sha256": source_hashes,
        "accepted_control_manifest_is_bound": True,
        "exact_champion_reconstructed_from_three_components": True,
        "exact_champion_control_identity": {str(item["cutoff"]): item for item in controls},
        "canonical_views": {name: [list(x) for x in segments] for name, segments in views.items()},
        "regression_gate_contract": dict(queue["evaluation"]["regression_gate"]),
        "support_group_contract": queue["evaluation"]["support_groups"],
        "oof_row_cutoff_mapping": mapping_audit,
        "exact_score_controls": controls,
        "early_fallback": {
            "view": "H12_E cycles56-67, cutoff55",
            "rows": int(len(early_rows)),
            "status": "EXACT_CHAMPION_FALLBACK",
            "identity_verified": True,
        },
        "fit_resource_budget": {
            "predictor_fits": 0, "preprocessing_fits": 0,
            "new_model_objects": 0, "official_fitting_started": False,
        },
        "synthetic_checks": {"status": "PASS", "checks": smoke},
        "no_test_or_sample_access": True,
        "no_future_score_labels_read": True,
        "oof_target_arrays_opened": False,
        "no_score_label_passed_to_adapter": True,
        "no_candidate_prediction_or_metric_computed": True,
        "no_r18_artifact_files_written": True,
        "python_version": platform.python_version(),
        "preflight_identity_sha256": json_sha256({
            "card": dict(card), "input_sha256": input_hashes,
            "source_sha256": source_hashes, "views": views,
            "fit_count": 0,
        }),
    }
    state = {
        "train_path": train_path,
        "frame": frame,
        "cycles": cycles,
        "history_targets": history_targets,
        "oof": oof,
        "controls_by_cutoff": controls_by_cutoff,
        "mapping_audit": mapping_audit,
        "inputs": inputs,
        "source_hashes": source_hashes,
    }
    return preflight, state


def _preflight(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    result, _ = _prepare_preflight(card, queue)
    return result


def _freeze_identity(card: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Build a score-blind identity: no candidate deltas or outcome metrics."""
    recipe = {
        "policy": "R_CALIBRATE",
        "change": "delta=clip(0.5*median(y-p),-2,2) using four complete strictly-past exact-champion OOF cycles",
        "residual_sign": "y - p; positive residual raises future prediction",
        "history_by_cutoff": {
            "67": {"cycles": [64, 65, 66, 67], "source_cutoff": 63},
            "75": {"cycles": [72, 73, 74, 75], "source_cutoff": 71},
        },
        "fixed_h12_l": "cutoff67 correction from cycles64-67 is held fixed for cycles68-79",
        "score_clip": [0.0, 100.0],
        "cutoff55_and_early": "exact champion fallback; no offset",
        "exact_champion": {
            "lgb_svd32": 0.5625,
            "hybrid_ridge": 0.1875,
            "sparse_lgb_rank50": 0.25,
            "sparse_rank50": "0.5 raw + 0.5 within-cycle average-rank percentile, clipped [0,100]",
        },
        "predictor_fits": 0,
        "preprocessing_fits": 0,
    }
    return {
        "schema_version": 1,
        "card_id": CARD_ID,
        "card": dict(card),
        "recipe": recipe,
        "input_sha256": dict(preflight["input_sha256"]),
        "source_sha256": dict(preflight["source_sha256"]),
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": importlib_metadata.version("numpy"),
            "pandas": importlib_metadata.version("pandas"),
            "scipy": importlib_metadata.version("scipy"),
            "threadpoolctl": importlib_metadata.version("threadpoolctl"),
        },
        "evaluation_views": dict(preflight["canonical_views"]),
        "evaluation_gate_contract": dict(preflight["regression_gate_contract"]),
        "support_group_contract": preflight["support_group_contract"],
        "resource_contract": dict(preflight["fit_resource_budget"]),
        "control_identity": dict(preflight["exact_champion_control_identity"]),
        "oof_history_support": list(preflight["oof_row_cutoff_mapping"]["calibration_history_support"]),
        "accepted_manifest_sha256": preflight["accepted_manifest_sha256"],
        "control_lineage": dict(preflight["frozen_oof_lineage"]),
        "output_contract": {
            "freeze": str(FREEZE_PATH.relative_to(ROOT)),
            "fit_receipts": str(RUN_DIR.relative_to(ROOT)),
            "report": str(REPORT_DIR.relative_to(ROOT)),
            "completion_marker": str(PUBLICATION_PATH.relative_to(ROOT)),
        },
        "validation_labels_passed_to_adapter": False,
        "future_target_values_in_freeze": False,
        "candidate_delta_in_freeze": False,
        "test_or_sample_used": False,
    }


def _freeze_document(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "card_id": CARD_ID,
        "identity": dict(identity),
        "signature": json_sha256(identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _verify_freeze_document(document: Mapping[str, Any], expected_identity: Mapping[str, Any]) -> str:
    identity = document.get("identity")
    if (document.get("schema_version") != 1 or document.get("card_id") != CARD_ID
            or not isinstance(identity, Mapping)):
        raise BatchIntegrityError("R18 freeze document schema/card identity is invalid")
    signature = str(document.get("signature", ""))
    if not signature or signature != json_sha256(identity):
        raise BatchIntegrityError("R18 freeze signature does not match its persisted identity")
    if dict(identity) != dict(expected_identity):
        raise BatchIntegrityError("Current R18 sources, inputs, recipe, views, controls, or dependencies differ from freeze")
    return signature


def _assert_run_outputs_absent() -> None:
    existing = [str(path) for path in (RUN_DIR, REPORT_DIR) if path.exists()]
    if existing:
        raise BatchIntegrityError(
            "R18 output directory already exists; refusing overwrite or partial resume: " + ", ".join(existing)
        )


def _freeze_only(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    preflight, _ = _prepare_preflight(card, queue)
    _assert_run_outputs_absent()
    identity = _freeze_identity(card, preflight)
    document = _freeze_document(identity)
    if FREEZE_PATH.exists():
        current = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        signature = _verify_freeze_document(current, identity)
        return {
            "card_id": CARD_ID, "status": "FROZEN_ALREADY_VERIFIED",
            "freeze_path": str(FREEZE_PATH.relative_to(ROOT)), "signature": signature,
            "freeze_sha256": file_sha256(FREEZE_PATH),
            "candidate_delta_computed": False, "candidate_metrics_computed": False,
            "future_target_values_in_freeze": False,
        }
    write_json_exclusive(FREEZE_PATH, document)
    return {
        "card_id": CARD_ID, "status": "FROZEN_BEFORE_R18_DELTA",
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)), "signature": document["signature"],
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "candidate_delta_computed": False, "candidate_metrics_computed": False,
        "future_target_values_in_freeze": False,
    }


def _read_score_targets_after_freeze(
    train_path: Path, cycles: np.ndarray, expected_train_sha256: str,
) -> dict[int, float]:
    """Read only declared Train score-cycle targets, after freeze verification."""
    if file_sha256(train_path) != expected_train_sha256:
        raise BatchIntegrityError("Official Train changed after frozen-signature verification")
    header = pd.read_csv(train_path, nrows=0).columns.tolist()
    cycle_index, target_index = header.index("cycle_code"), header.index("threat_rank")
    allowed_cycles = set(range(56, 80))
    values: dict[int, float] = {}
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row_position, record in enumerate(reader):
            cycle = int(record[cycle_index])
            if cycle in allowed_cycles:
                value = float(record[target_index])
                if not np.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise BatchIntegrityError(f"Invalid post-freeze score target at row {row_position}")
                values[row_position] = value
    expected_rows = np.flatnonzero(np.isin(cycles, np.asarray(sorted(allowed_cycles), dtype=np.int64)))
    if sorted(values) != expected_rows.tolist():
        raise BatchIntegrityError("Post-freeze score-target rows differ from the frozen LATE/H12_E population")
    if file_sha256(train_path) != expected_train_sha256:
        raise BatchIntegrityError("Official Train changed while post-freeze score labels were read")
    return values


def _history_for_cutoff(cutoff: int, state: Mapping[str, Any], score_cycles: np.ndarray) -> dict[str, np.ndarray]:
    frozen = POLICY_BY_CUTOFF.get(int(cutoff))
    if frozen is None:
        raise BatchIntegrityError(f"No R18 past-OOF history is defined for cutoff {cutoff}")
    oof = state["oof"]
    rows = np.asarray(oof["primary_rows"], dtype=np.int64)
    cycles = np.asarray(oof["primary_cycle"], dtype=np.int64)
    selected = np.flatnonzero(np.isin(cycles, np.asarray(frozen["history_cycles"], dtype=np.int64)))
    history_rows = rows[selected]
    history_cycles = cycles[selected]
    if len(history_rows) < MIN_HISTORY_ROWS or len(np.unique(history_rows)) != len(history_rows):
        raise BatchIntegrityError("Frozen R18 history support is insufficient or duplicated")
    if tuple(sorted(np.unique(history_cycles).tolist())) != tuple(frozen["history_cycles"]):
        raise BatchIntegrityError("Frozen R18 history does not contain all four expected cycles")
    source = np.full(len(history_rows), int(frozen["source_cutoff"]), dtype=np.int64)
    if np.any(source >= history_cycles) or np.any(history_cycles >= int(np.min(score_cycles))):
        raise BatchIntegrityError("R18 history contains same/future-cycle labels relative to score view")
    target_map = state["history_targets"]
    try:
        history_target = np.asarray([target_map[int(row)] for row in history_rows], dtype=np.float64)
    except KeyError as exc:
        raise BatchIntegrityError(f"Allowed past OOF target is missing for Train row {exc.args[0]}") from exc
    prediction = np.asarray(oof["primary_prediction"], dtype=np.float64)[selected]
    return {
        "rows": history_rows,
        "cycle": history_cycles,
        "source_cutoff": source,
        "target": history_target,
        "prediction": prediction,
    }


def _frozen_candidate_blocks(state: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    controls_by_cutoff = state["controls_by_cutoff"]
    output: dict[int, dict[str, Any]] = {}
    for cutoff in (55, 67, 75):
        control = controls_by_cutoff[cutoff]
        score_cycles = np.asarray(control["cycle"], dtype=np.int64)
        model_controls: dict[str, Any] = {
            "baseline_prediction": np.asarray(control["baseline_prediction"], dtype=np.float64),
            "score_cycle": score_cycles,
        }
        history = None
        if cutoff in POLICY_BY_CUTOFF:
            history = _history_for_cutoff(cutoff, state, score_cycles)
            model_controls.update({
                "history_rows": history["rows"],
                "history_cycle": history["cycle"],
                "history_source_cutoff": history["source_cutoff"],
                "history_target": history["target"],
                "history_prediction": history["prediction"],
            })
        result = r18_fit_predict(
            fit_x=None, fit_y=None, score_x=None, controls=model_controls,
            card_id=CARD_ID, cutoff=cutoff,
        )
        prediction = np.asarray(result["prediction"], dtype=np.float64)
        if prediction.shape != score_cycles.shape or not np.isfinite(prediction).all():
            raise BatchIntegrityError(f"R18 frozen candidate predictions are invalid at cutoff {cutoff}")
        output[cutoff] = {
            **control,
            "prediction": prediction,
            "history": history,
            "state": dict(result["state"]),
            "fit_predictor_count": int(result["fit_predictor_count"]),
        }
    return output


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(y, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or target.shape != predicted.shape or len(target) == 0:
        raise BatchIntegrityError("MAE inputs must be nonempty aligned vectors")
    if not np.isfinite(target).all() or not np.isfinite(predicted).all():
        raise BatchIntegrityError("MAE inputs must be finite")
    return float(np.mean(np.abs(predicted - target)))


def _slice_block(block: Mapping[str, Any], cycles: np.ndarray, low: int, high: int) -> dict[str, np.ndarray]:
    rows = np.asarray(block["rows"], dtype=np.int64)
    row_cycles = cycles[rows]
    selected = (row_cycles >= int(low)) & (row_cycles <= int(high))
    return {
        "rows": rows[selected],
        "cycle": row_cycles[selected],
        "cutoff": np.full(int(selected.sum()), int(block["cutoff"]), dtype=np.int64),
        "baseline": np.asarray(block["baseline_prediction"], dtype=np.float64)[selected],
        "candidate": np.asarray(block["prediction"], dtype=np.float64)[selected],
    }


def _join_slices(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    combined = {key: np.concatenate((np.asarray(left[key]), np.asarray(right[key])))
                for key in ("rows", "cycle", "cutoff", "baseline", "candidate")}
    if len(np.unique(combined["rows"])) != len(combined["rows"]):
        raise BatchIntegrityError("R18 pooled primary view contains duplicate Train rows")
    return combined


def _entry_friction_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "missing"


def _group_labels_for_view(
    *, view: Mapping[str, np.ndarray], frame: pd.DataFrame, target: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(view["rows"], dtype=np.int64)
    cutoffs = np.asarray(view["cutoff"], dtype=np.int64)
    y = np.asarray(target, dtype=np.float64)
    labels: dict[str, np.ndarray] = {
        "entry_friction": np.asarray([
            _entry_friction_label(x) for x in frame.iloc[rows]["entry_friction"].tolist()
        ], dtype=object),
        "signal_stream": np.where(
            stream_missing_mask(frame.iloc[rows]["signal_stream"]), "missing", "present"
        ).astype(object),
        "target_tail": np.select(
            [y <= 5.0, y >= 95.0],
            ["low_le5", "high_ge95"], default="middle",
        ).astype(object),
    }
    for column in ("org_code", "module_code", "pattern_code"):
        group = np.full(len(rows), "", dtype=object)
        for cutoff in np.unique(cutoffs):
            positions = np.flatnonzero(cutoffs == cutoff)
            fit_mask = np.asarray(frame["cycle_code"], dtype=np.int64) <= int(cutoff)
            missing, unseen, _ = category_novelty_flags(
                frame.loc[fit_mask, column].tolist(),
                frame.iloc[rows[positions]][column].tolist(),
            )
            group[positions] = np.where(missing, "missing", np.where(unseen, "unseen", "seen"))
        labels[f"{column}_status"] = group
    return labels


def _subgroup_table(
    *, view_name: str, view: Mapping[str, np.ndarray], frame: pd.DataFrame,
    target_by_row: Mapping[int, float], support_min: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = np.asarray(view["rows"], dtype=np.int64)
    y = np.asarray([target_by_row[int(row)] for row in rows], dtype=np.float64)
    labels_by_field = _group_labels_for_view(view=view, frame=frame, target=y)
    result: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for field, labels in labels_by_field.items():
        categories = sorted(set(str(x) for x in labels))
        if field == "entry_friction":
            categories = sorted(set(categories) | {"0", "1", "missing"})
        elif field == "signal_stream":
            categories = ["present", "missing"]
        elif field == "target_tail":
            categories = ["low_le5", "middle", "high_ge95"]
        elif field.endswith("_status"):
            categories = ["seen", "unseen", "missing"]
        field_rows: dict[str, Any] = {}
        for group in categories:
            mask = labels == group
            n = int(mask.sum())
            supported = n >= int(support_min)
            if n:
                candidate_mae = _mae(y[mask], np.asarray(view["candidate"])[mask])
                baseline_mae = _mae(y[mask], np.asarray(view["baseline"])[mask])
                harm = float(candidate_mae - baseline_mae)
            else:
                candidate_mae = baseline_mae = harm = None
            guard = bool(
                supported and view_name in {"W4", "W6", "H12_L"}
                and ((field == "target_tail" and group in {"low_le5", "high_ge95"})
                     or (field == "signal_stream" and group == "missing"))
            )
            passed = None if not guard else bool(harm <= 0.25)
            field_rows[group] = {
                "n": n, "supported": supported,
                "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                "candidate_minus_baseline_mae": harm,
                "guard_applies": guard, "guard_pass": passed,
            }
            if guard and not passed:
                failures.append({"view": view_name, "field": field, "group": group,
                                 "n": n, "candidate_minus_baseline_mae": harm})
        result[field] = field_rows
    return result, failures


def _evaluate_frozen_predictions(
    *, blocks: Mapping[int, Mapping[str, Any]], cycles: np.ndarray, frame: pd.DataFrame,
    target_by_row: Mapping[int, float],
) -> dict[str, Any]:
    views: dict[str, dict[str, np.ndarray]] = {
        "W4": _slice_block(blocks[67], cycles, 68, 71),
        "W6": _slice_block(blocks[75], cycles, 76, 79),
        "H12_L": _slice_block(blocks[67], cycles, 68, 79),
        "H12_E": _slice_block(blocks[55], cycles, 56, 67),
    }
    views["primary"] = _join_slices(views["W4"], views["W6"])
    view_results: dict[str, Any] = {}
    subgroup_results: dict[str, Any] = {}
    subgroup_failures: list[dict[str, Any]] = []
    for name, view in views.items():
        rows = np.asarray(view["rows"], dtype=np.int64)
        if len(np.unique(rows)) != len(rows) or not len(rows):
            raise BatchIntegrityError(f"R18 {name} row view is empty or duplicated")
        y = np.asarray([target_by_row[int(row)] for row in rows], dtype=np.float64)
        base_mae = _mae(y, np.asarray(view["baseline"], dtype=np.float64))
        candidate_mae = _mae(y, np.asarray(view["candidate"], dtype=np.float64))
        item = {
            "rows": int(len(rows)),
            "cycle_min": int(np.min(view["cycle"])),
            "cycle_max": int(np.max(view["cycle"])),
            "baseline_mae": base_mae,
            "candidate_mae": candidate_mae,
            "candidate_minus_baseline_mae": float(candidate_mae - base_mae),
            "gain_positive_is_better": float(base_mae - candidate_mae),
        }
        if name == "H12_E":
            item["policy"] = "exact champion fallback"
        view_results[name] = item
        if name != "H12_E":
            subgroups, failures = _subgroup_table(
                view_name=name, view=view, frame=frame,
                target_by_row=target_by_row, support_min=100,
            )
            subgroup_results[name] = subgroups
            subgroup_failures.extend(failures)

    gates = {
        "late_primary_gain_min_0_03": view_results["primary"]["gain_positive_is_better"] >= 0.03,
        "H12_L_harm_max_0_05": view_results["H12_L"]["candidate_minus_baseline_mae"] <= 0.05,
        "W4_individual_harm_max_0_5": view_results["W4"]["candidate_minus_baseline_mae"] <= 0.5,
        "W6_individual_harm_max_0_5": view_results["W6"]["candidate_minus_baseline_mae"] <= 0.5,
        "supported_tail_or_missing_harm_max_0_25": len(subgroup_failures) == 0,
    }
    return {
        "evaluation_profile": "LATE; W4+W6 primary and H12_L are overlapping temporal development views",
        "views": view_results,
        "subgroups": subgroup_results,
        "subgroup_gate_failures": subgroup_failures,
        "support_rule": "n>=100; smaller groups descriptive/unsupported",
        "gates": gates,
        "all_pilot_gates_pass": bool(all(gates.values())),
        "verdict": "ADVANCE" if all(gates.values()) else "RETIRE",
        "promotion_limit": "late-only pilot; no external inference/submission promotion",
        "early_fallback_reported_separately": True,
    }


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _assert_current_frozen_sources(identity: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    current_inputs = {name: file_sha256(path) for name, path in sorted(state["inputs"].items())}
    current_sources = {str(path.relative_to(ROOT)): file_sha256(path) for path in _source_files()}
    if current_inputs != dict(identity["input_sha256"]) or current_sources != dict(identity["source_sha256"]):
        raise BatchIntegrityError("A frozen source/input changed during R18 run; no result publication is allowed")


def _independent_mae(target: Any, prediction: Any) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise BatchIntegrityError("Independent MAE verifier received invalid vectors")
    return float(np.abs(y - p).sum(dtype=np.float64) / y.size)


def _verify_subgroup_table(
    *, view_name: str, view: Mapping[str, np.ndarray], frame: pd.DataFrame,
    target: Any, stored: Mapping[str, Any], support_min: int = 100,
) -> dict[str, Any]:
    """Reconstruct every published subgroup denominator and MAE independently."""
    rows = np.asarray(view["rows"], dtype=np.int64)
    y = np.asarray(target, dtype=np.float64)
    if y.shape != rows.shape or not np.isfinite(y).all():
        raise BatchIntegrityError(f"R18 {view_name} subgroup targets are invalid or misaligned")
    labels_by_field = _group_labels_for_view(view=view, frame=frame, target=y)
    if not isinstance(stored, Mapping) or set(stored) != set(labels_by_field):
        raise BatchIntegrityError(f"R18 {view_name} subgroup fields differ from the published table")

    checked_groups = 0
    for field, labels in labels_by_field.items():
        categories = sorted(set(str(x) for x in labels))
        if field == "entry_friction":
            categories = sorted(set(categories) | {"0", "1", "missing"})
        elif field == "signal_stream":
            categories = ["present", "missing"]
        elif field == "target_tail":
            categories = ["low_le5", "middle", "high_ge95"]
        elif field.endswith("_status"):
            categories = ["seen", "unseen", "missing"]
        stored_field = stored.get(field)
        if not isinstance(stored_field, Mapping) or set(stored_field) != set(categories):
            raise BatchIntegrityError(f"R18 {view_name} subgroup categories differ for {field}")

        for group in categories:
            selected = labels == group
            n = int(selected.sum())
            supported = n >= int(support_min)
            if n:
                candidate_mae = _independent_mae(y[selected], np.asarray(view["candidate"])[selected])
                baseline_mae = _independent_mae(y[selected], np.asarray(view["baseline"])[selected])
                harm = float(candidate_mae - baseline_mae)
            else:
                candidate_mae = baseline_mae = harm = None
            guard = bool(
                supported and view_name in {"W4", "W6", "H12_L"}
                and ((field == "target_tail" and group in {"low_le5", "high_ge95"})
                     or (field == "signal_stream" and group == "missing"))
            )
            guard_pass = None if not guard else bool(harm <= 0.25)
            actual = stored_field[group]
            expected = {
                "n": n, "supported": supported,
                "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                "candidate_minus_baseline_mae": harm,
                "guard_applies": guard, "guard_pass": guard_pass,
            }
            if not isinstance(actual, Mapping) or set(actual) != set(expected):
                raise BatchIntegrityError(
                    f"R18 {view_name} subgroup result schema differs for {field}={group}"
                )
            for key, value in expected.items():
                observed = actual[key]
                if value is None:
                    matches = observed is None
                elif isinstance(value, (int, bool)):
                    matches = observed == value and type(observed) is type(value)
                else:
                    matches = (observed is not None and np.isclose(
                        float(observed), float(value), atol=1e-12, rtol=0.0))
                if not matches:
                    raise BatchIntegrityError(
                        f"R18 {view_name} subgroup {field}={group} {key} differs: "
                        f"stored={observed!r}, reconstructed={value!r}"
                    )
            checked_groups += 1
    return {"status": "PASS", "fields": len(labels_by_field), "groups": checked_groups}


def _independent_verify_receipts(
    *, signature: str, state: Mapping[str, Any], blocks: Mapping[int, Mapping[str, Any]],
    target_by_row: Mapping[int, float], results: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read persisted bundles and independently recheck outputs and metrics."""
    frame = state["frame"]
    cycles = np.asarray(state["cycles"], dtype=np.int64)
    loaded: dict[int, dict[str, Any]] = {}
    bundle_checks = []
    for cutoff in (55, 67, 75):
        memory = blocks[cutoff]
        rows = np.asarray(memory["rows"], dtype=np.int64)
        receipt_path = RUN_DIR / f"R18_fit{cutoff}.npz"
        expected_identity = {
            "freeze_signature": signature, "card_id": CARD_ID,
            "cutoff": cutoff, "score_rows": rows.tolist(),
        }
        arrays, sidecar = read_bundle(receipt_path, expected_identity)
        meta = sidecar["metadata"]
        if meta.get("fit_predictor_count") != 0 or meta.get("test_data_used") is not False:
            raise BatchIntegrityError(f"R18 fit receipt metadata is invalid at cutoff {cutoff}")
        for key, expected in (
            ("score_rows", rows),
            ("score_cycles", cycles[rows]),
            ("baseline_prediction", memory["baseline_prediction"]),
            ("prediction", memory["prediction"]),
        ):
            if key not in arrays or not np.array_equal(arrays[key], np.asarray(expected)):
                raise BatchIntegrityError(f"R18 persisted receipt {key} differs at cutoff {cutoff}")
        score_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)
        if not np.array_equal(arrays.get("score_record_ids"), score_ids):
            raise BatchIntegrityError(f"R18 score record_id alignment differs at cutoff {cutoff}")
        state_meta = meta.get("state", {})
        delta = float(state_meta.get("delta", 0.0))
        if cutoff in POLICY_BY_CUTOFF:
            history = memory["history"]
            for key, expected in (
                ("history_rows", history["rows"]),
                ("history_cycles", history["cycle"]),
                ("history_source_cutoff", history["source_cutoff"]),
                ("history_prediction", history["prediction"]),
            ):
                if key not in arrays or not np.array_equal(arrays[key], np.asarray(expected)):
                    raise BatchIntegrityError(f"R18 persisted {key} differs from frozen history at cutoff {cutoff}")
            expected_delta = float(np.clip(
                0.5 * np.median(history["target"] - history["prediction"]), -2.0, 2.0
            ))
            if not np.isclose(delta, expected_delta, atol=0.0, rtol=0.0):
                raise BatchIntegrityError(f"R18 persisted delta fails independent history reconstruction at {cutoff}")
            expected_prediction = np.clip(
                np.asarray(arrays["baseline_prediction"]) + expected_delta, 0.0, 100.0
            )
        else:
            if delta != 0.0 or not np.array_equal(arrays["prediction"], arrays["baseline_prediction"]):
                raise BatchIntegrityError("R18 cutoff55 fallback is not an exact-champion identity")
            expected_prediction = np.asarray(arrays["baseline_prediction"])
        if not np.array_equal(arrays["prediction"], expected_prediction):
            raise BatchIntegrityError(f"R18 receipt prediction formula differs at cutoff {cutoff}")
        loaded[cutoff] = {
            "cutoff": cutoff, "rows": arrays["score_rows"],
            "baseline_prediction": arrays["baseline_prediction"],
            "prediction": arrays["prediction"],
        }
        bundle_checks.append({
            "cutoff": cutoff, "bundle_readback": True,
            "row_and_id_alignment": True, "prediction_reconstruction": True,
            "fit_predictor_count": 0,
        })

    view_specs = {
        "W4": (loaded[67], 68, 71),
        "W6": (loaded[75], 76, 79),
        "H12_L": (loaded[67], 68, 79),
        "H12_E": (loaded[55], 56, 67),
    }
    metric_checks = {}
    for name, (block, low, high) in view_specs.items():
        rows = np.asarray(block["rows"], dtype=np.int64)
        mask = (cycles[rows] >= low) & (cycles[rows] <= high)
        selected_rows = rows[mask]
        y = np.asarray([target_by_row[int(row)] for row in selected_rows], dtype=np.float64)
        baseline_mae = _independent_mae(y, np.asarray(block["baseline_prediction"])[mask])
        candidate_mae = _independent_mae(y, np.asarray(block["prediction"])[mask])
        frozen_result = results["views"][name]
        if (not np.isclose(baseline_mae, float(frozen_result["baseline_mae"]), atol=1e-12, rtol=0.0)
                or not np.isclose(candidate_mae, float(frozen_result["candidate_mae"]), atol=1e-12, rtol=0.0)):
            raise BatchIntegrityError(f"Independent MAE differs from stored result for {name}")
        metric_checks[name] = {
            "rows": int(len(selected_rows)), "baseline_mae": baseline_mae,
            "candidate_mae": candidate_mae, "match": True,
        }

    primary_slices = []
    for cutoff, low, high in ((67, 68, 71), (75, 76, 79)):
        block = loaded[cutoff]
        rows = np.asarray(block["rows"], dtype=np.int64)
        mask = (cycles[rows] >= low) & (cycles[rows] <= high)
        primary_slices.append((rows[mask], np.asarray(block["baseline_prediction"])[mask],
                               np.asarray(block["prediction"])[mask]))
    primary_rows = np.concatenate([item[0] for item in primary_slices])
    if len(np.unique(primary_rows)) != len(primary_rows):
        raise BatchIntegrityError("Independent W4+W6 primary reconstruction contains duplicate rows")
    primary_target = np.asarray([target_by_row[int(row)] for row in primary_rows], dtype=np.float64)
    primary_baseline = np.concatenate([item[1] for item in primary_slices])
    primary_candidate = np.concatenate([item[2] for item in primary_slices])
    primary_baseline_mae = _independent_mae(primary_target, primary_baseline)
    primary_candidate_mae = _independent_mae(primary_target, primary_candidate)
    stored_primary = results["views"]["primary"]
    if (not np.isclose(primary_baseline_mae, float(stored_primary["baseline_mae"]), atol=1e-12, rtol=0.0)
            or not np.isclose(primary_candidate_mae, float(stored_primary["candidate_mae"]), atol=1e-12, rtol=0.0)):
        raise BatchIntegrityError("Independent W4+W6 primary MAE differs from stored result")
    metric_checks["primary"] = {
        "rows": int(len(primary_rows)), "baseline_mae": primary_baseline_mae,
        "candidate_mae": primary_candidate_mae, "match": True,
        "reconstructed_from": ["W4", "W6"],
    }

    subgroup_failures = []
    subgroup_table_checks = {}
    for name in ("W4", "W6", "H12_L"):
        block, low, high = view_specs[name]
        rows = np.asarray(block["rows"], dtype=np.int64)
        mask = (cycles[rows] >= low) & (cycles[rows] <= high)
        rows = rows[mask]
        y = np.asarray([target_by_row[int(row)] for row in rows], dtype=np.float64)
        group_view = {
            "rows": rows, "cycle": cycles[rows],
            "cutoff": np.full(len(rows), int(block["cutoff"]), dtype=np.int64),
            "baseline": np.asarray(block["baseline_prediction"])[mask],
            "candidate": np.asarray(block["prediction"])[mask],
        }
        labels = _group_labels_for_view(view=group_view, frame=frame, target=y)
        subgroup_table_checks[name] = _verify_subgroup_table(
            view_name=name, view=group_view, frame=frame, target=y,
            stored=results.get("subgroups", {}).get(name), support_min=100,
        )
        guard_groups = (
            ("target_tail", "low_le5"), ("target_tail", "high_ge95"),
            ("signal_stream", "missing"),
        )
        for field, group in guard_groups:
            selected = labels[field] == group
            n = int(selected.sum())
            if n >= 100:
                harm = _independent_mae(y[selected], group_view["candidate"][selected]) - _independent_mae(
                    y[selected], group_view["baseline"][selected]
                )
                if harm > 0.25:
                    subgroup_failures.append({
                        "view": name, "field": field, "group": group,
                        "n": n, "candidate_minus_baseline_mae": float(harm),
                    })
    if subgroup_failures:
        raise BatchIntegrityError(f"Independent subgroup guard found failures: {subgroup_failures}")
    independent_gates = {
        "late_primary_gain_min_0_03": primary_baseline_mae - primary_candidate_mae >= 0.03,
        "H12_L_harm_max_0_05": metric_checks["H12_L"]["candidate_mae"]
            - metric_checks["H12_L"]["baseline_mae"] <= 0.05,
        "W4_individual_harm_max_0_5": metric_checks["W4"]["candidate_mae"]
            - metric_checks["W4"]["baseline_mae"] <= 0.5,
        "W6_individual_harm_max_0_5": metric_checks["W6"]["candidate_mae"]
            - metric_checks["W6"]["baseline_mae"] <= 0.5,
        "supported_tail_or_missing_harm_max_0_25": True,
    }
    if independent_gates != dict(results["gates"]):
        raise BatchIntegrityError("Independent metric gates differ from the result gate receipt")
    if bool(all(independent_gates.values())) != bool(results["all_pilot_gates_pass"]):
        raise BatchIntegrityError("Independent overall pilot decision differs from result status")
    return {
        "status": "PASS",
        "receipt_checks": bundle_checks,
        "view_metric_checks": metric_checks,
        "subgroup_table_checks": subgroup_table_checks,
        "independent_supported_tail_or_missing_guard_failures": subgroup_failures,
        "independent_gate_reconstruction": independent_gates,
        "test_or_sample_used": False,
    }


def _run_frozen_locked(
    card: Mapping[str, Any], queue: Mapping[str, Any], freeze_document: Mapping[str, Any],
) -> dict[str, Any]:
    preflight, state = _prepare_preflight(card, queue)
    identity = _freeze_identity(card, preflight)
    signature = _verify_freeze_document(freeze_document, identity)
    freeze_sha256 = file_sha256(FREEZE_PATH)
    _assert_run_outputs_absent()
    _assert_current_frozen_sources(identity, state)

    # Only the two card-approved past target blocks are passed to the adapter.
    # Future score targets are opened after the persisted freeze is verified.
    blocks = _frozen_candidate_blocks(state)
    target_by_row = _read_score_targets_after_freeze(
        state["train_path"], state["cycles"], identity["input_sha256"]["official_regression_train"],
    )
    evaluation = _evaluate_frozen_predictions(
        blocks=blocks, cycles=state["cycles"], frame=state["frame"], target_by_row=target_by_row,
    )
    calibration = {
        str(cutoff): dict(blocks[cutoff]["state"])
        for cutoff in (67, 75)
    }
    results = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "task": "regression",
        "status": "PILOT_COMPUTED_AWAITING_INDEPENDENT_QA",
        "freeze_signature": signature,
        "freeze_sha256": freeze_sha256,
        "input_sha256": dict(identity["input_sha256"]),
        "source_sha256": dict(identity["source_sha256"]),
        "fit_predictor_count": 0,
        "preprocessing_fit_count": 0,
        "calibration": calibration,
        "views": evaluation["views"],
        "subgroups": evaluation["subgroups"],
        "subgroup_gate_failures": evaluation["subgroup_gate_failures"],
        "support_rule": evaluation["support_rule"],
        "gates": evaluation["gates"],
        "all_pilot_gates_pass": evaluation["all_pilot_gates_pass"],
        "verdict": evaluation["verdict"],
        "promotion_limit": evaluation["promotion_limit"],
        "fallback_coverage": {
            "H12_E": {
                "rows": int(len(blocks[55]["rows"])),
                "exact_champion_rows": int(len(blocks[55]["rows"])),
                "coverage": 1.0,
                "gate_applies": False,
            },
            "late_supported_views": {
                "W4_rows": int(evaluation["views"]["W4"]["rows"]),
                "W6_rows": int(evaluation["views"]["W6"]["rows"]),
                "H12_L_rows": int(evaluation["views"]["H12_L"]["rows"]),
                "history_fallback_rows": 0,
            },
        },
        "test_or_sample_used": False,
        "future_score_labels_passed_to_adapter": False,
        "outcome_reuse_warning": "Temporal development outcomes are reused; this late-only pilot is not an untouched holdout.",
    }

    _assert_run_outputs_absent()
    receipt_paths: list[Path] = []
    for cutoff in (55, 67, 75):
        block = blocks[cutoff]
        rows = np.asarray(block["rows"], dtype=np.int64)
        arrays: dict[str, np.ndarray] = {
            "score_rows": rows,
            "score_cycles": np.asarray(block["cycle"], dtype=np.int64),
            "score_record_ids": np.asarray(block["record_id"], dtype=str),
            "baseline_prediction": np.asarray(block["baseline_prediction"], dtype=np.float64),
            "prediction": np.asarray(block["prediction"], dtype=np.float64),
            "delta": np.full(len(rows), float(block["state"]["delta"]), dtype=np.float64),
        }
        if block["history"] is not None:
            history = block["history"]
            arrays.update({
                "history_rows": np.asarray(history["rows"], dtype=np.int64),
                "history_cycles": np.asarray(history["cycle"], dtype=np.int64),
                "history_source_cutoff": np.asarray(history["source_cutoff"], dtype=np.int64),
                "history_prediction": np.asarray(history["prediction"], dtype=np.float64),
            })
        receipt_path = RUN_DIR / f"R18_fit{cutoff}.npz"
        receipt_identity = {
            "freeze_signature": signature, "card_id": CARD_ID,
            "cutoff": cutoff, "score_rows": rows.tolist(),
        }
        write_bundle_exclusive(
            receipt_path, arrays, receipt_identity,
            {
                "card_id": CARD_ID, "cutoff": cutoff,
                "fit_predictor_count": 0, "preprocessing_fit_count": 0,
                "fit_rows_used_by_predictor": 0, "score_rows": int(len(rows)),
                "state": dict(block["state"]),
                "test_data_used": False, "score_labels_passed_to_adapter": False,
            },
        )
        receipt_paths.extend([receipt_path, receipt_path.with_suffix(".json")])

    verification = _independent_verify_receipts(
        signature=signature, state=state, blocks=blocks,
        target_by_row=target_by_row, results=results,
    )
    results["status"] = "PILOT_VERIFIED"
    results["independent_verification"] = verification["status"]
    _assert_current_frozen_sources(identity, state)
    if file_sha256(FREEZE_PATH) != freeze_sha256:
        raise BatchIntegrityError("R18 freeze changed during run; refusing publication")

    result_path = REPORT_DIR / "results.json"
    verification_path = REPORT_DIR / "verification.json"
    summary_path = REPORT_DIR / "summary.md"
    write_json_exclusive(result_path, results)
    write_json_exclusive(verification_path, {
        "card_id": CARD_ID, "status": verification["status"],
        "freeze_signature": signature, "freeze_sha256": freeze_sha256,
        "checks": verification,
    })
    _write_text_exclusive(summary_path, _render_summary(results, signature))
    published_files = [*receipt_paths, result_path, verification_path, summary_path]
    manifest = {
        "schema_version": 1,
        "card_id": CARD_ID,
        "status": "COMPLETE",
        "freeze_signature": signature,
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in sorted(published_files, key=str)
        },
        "results_sha256": file_sha256(result_path),
        "verification_sha256": file_sha256(verification_path),
        "summary_sha256": file_sha256(summary_path),
    }
    write_json_exclusive(PUBLICATION_PATH, manifest)
    return {
        "card_id": CARD_ID, "status": "PILOT_VERIFIED",
        "freeze_signature": signature,
        "results_path": str(result_path.relative_to(ROOT)),
        "verification_path": str(verification_path.relative_to(ROOT)),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "publication_path": str(PUBLICATION_PATH.relative_to(ROOT)),
        "verdict": results["verdict"],
        "all_pilot_gates_pass": results["all_pilot_gates_pass"],
        "fit_predictor_count": 0,
    }


def _run_frozen(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card_and_queue(card, queue)
    if not FREEZE_PATH.is_file():
        raise BatchIntegrityError("R18 --run-frozen requires the prior --freeze-only artifact")
    document = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    raw_identity = document.get("identity")
    if (document.get("card_id") != CARD_ID or not isinstance(raw_identity, Mapping)
            or document.get("signature") != json_sha256(raw_identity)):
        raise BatchIntegrityError("Persisted R18 freeze document is malformed or has a bad signature")
    lock_path = RUN_DIR.parent / ".training.lock"
    with exclusive_training_slot(lock_path):
        return _run_frozen_locked(card, queue, document)


def _render_summary(results: Mapping[str, Any], signature: str) -> str:
    views = results["views"]
    lines = [
        "# R18 — Recent OOF median location offset",
        "",
        f"Status: **{results['verdict']}** (late-only pilot; no external promotion).",
        f"Freeze signature: {signature}",
        "",
        "| View | Rows | Baseline MAE | Candidate MAE | Gain (positive is better) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("primary", "W4", "W6", "H12_L", "H12_E"):
        item = views[name]
        lines.append(
            f"| {name} | {item['rows']} | {item['baseline_mae']:.6f} | "
            f"{item['candidate_mae']:.6f} | {item['gain_positive_is_better']:.6f} |"
        )
    lines.extend([
        "",
        "H12_E is reported separately and uses the exact-champion fallback.",
        "Candidate deltas use only the four complete past OOF cycles fixed in the R18 card.",
        "",
    ])
    return "\n".join(lines)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    if preflight_only:
        return _preflight(card, queue)
    raise BatchIntegrityError(
        "Use the R18 runner directly with exactly one of --preflight-only, --freeze-only, or --run-frozen"
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == CARD_ID)
    if args.preflight_only:
        result = _preflight(card, queue)
    elif args.freeze_only:
        result = _freeze_only(card, queue)
    else:
        result = _run_frozen(card, queue)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
