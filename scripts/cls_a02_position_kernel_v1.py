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
import tempfile
import time
from typing import Any
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.models.a02_position_kernel_v1 import (
    MISSING_THRESHOLD,
    PILOT_ID,
    SHUFFLE_SEED,
    assemble_full_champion,
    classification_metrics,
    construct_position_kernel,
    load_npz_allowlisted,
)


REPO_ROOT = SCRIPT_ROOT
OFFICIAL_ROOT = Path(os.environ.get("KASDAD_TP2_SOURCE_ROOT", REPO_ROOT.parent / "TP 2")).resolve()
REPORT_ROOT = REPO_ROOT / "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1"
PROTOCOL = REPO_ROOT / "experiments/optimization_v3/cls_a02_position_kernel_v1/protocol.json"
ROOT_PLAN = REPORT_ROOT / "ROOT_EXECUTION_PLAN.md"
ROOT_ADJUDICATION = REPORT_ROOT / "ROOT_INCIDENT_01_ADJUDICATION.md"
INCIDENT = REPORT_ROOT / "INCIDENT_01.json"
AMENDMENT = REPORT_ROOT / "AMENDMENT_01.md"
FREEZE = REPORT_ROOT / "FREEZE.json"
LEDGER = REPORT_ROOT / "BUDGET_LEDGER.jsonl"
RUN_ROOT = REPO_ROOT / "experiments/runs/simplification_v1/CLS_POST_BAG2_NEXT_V1"
TRAIN = OFFICIAL_ROOT / "data/raw/classification/Train.csv"
POS_ROOT = OFFICIAL_ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
POS_OFFSET_ROOT = OFFICIAL_ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
POISSON_PARENT_ROOT = REPO_ROOT / "experiments/CLS_POS_POISSON_BAG2_V1"
POISSON_PARENT_REPORT = REPO_ROOT / "reports/simplification_v1/CLS_POS_POISSON_BAG2_V1"
POISSON_PARENT_FREEZE = POISSON_PARENT_REPORT / "PREFIT_FREEZE.json"
POISSON_PARENT_ROOT_PASS = POISSON_PARENT_REPORT / "ROOT_PREFIT_PASS.json"
POISSON_PARENT_READY = POISSON_PARENT_ROOT / "PREDICTIONS_READY.json"
POISSON_PARENT_READBACK = POISSON_PARENT_ROOT / "READBACK_VERIFICATION.json"
ORIGINS = (67, 75)
ARMS = ("REAL", "SHUFFLED")
FIT_IDS = tuple(f"E{origin}_{arm}" for origin in ORIGINS for arm in ARMS)
EXPECTED_OFFICIAL_COMMIT = "7108c276c9904556a1d6eb186c46588ddf2ef6fb"
MAX_THREADS = 2
INITIAL_RAM_MIN = 1 * 1024**3
PROJECTED_RESERVE = 512 * 1024**2
ESTIMATED_ADDITIONAL = 512 * 1024**2

A02_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
    "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
    "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction",
    "correction_unclipped",
)
POS_MEMBERS = ("score_rows", "score_record_ids", "score_cycles", "present_mask")
POS_OFFSET_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "present_mask", "prob_offset_real",
)
POISSON_PARENT_MEMBERS = (
    "origin", "row_positions", "record_ids", "cycles", "present_mask", "p_a02",
    "p_pos_original", "p_champion", "label_champion", "matrix_manifest_sha256",
    "input_freeze_signature",
)
OUTPUT_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "missing_mask", "entry_friction",
    "text_load_group", "p_sparse25", "p_a02_original", "p_pos_original",
    "p_a02_candidate", "champion_score", "candidate_score", "champion_class",
    "candidate_class", "m0", "neighbor_count", "neighbor_source_rows", "neighbor_ids",
    "neighbor_kernel_similarity", "neighbor_weights", "position_kernel_positive_mass",
    "correction_raw", "correction", "mass_error", "neighbor_overlap_fraction",
)
FORBIDDEN_OUTPUT_MEMBERS = {"labels", "artifact_signal", "y_true", "target", "threat_rank"}


class PilotError(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise PilotError(f"Refusing to overwrite immutable JSON: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive and path.exists():
        tmp.unlink(missing_ok=True)
        raise PilotError(f"Immutable JSON appeared concurrently: {path}")
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray], *, exclusive: bool = True) -> None:
    keys = set(arrays)
    if keys != set(OUTPUT_MEMBERS):
        raise PilotError(f"Output schema mismatch: {sorted(keys ^ set(OUTPUT_MEMBERS))}")
    if keys & FORBIDDEN_OUTPUT_MEMBERS:
        raise PilotError("Output bundle contains a target-bearing member")
    if any(np.asarray(value).dtype.kind == "O" for value in arrays.values()):
        raise PilotError("Object arrays are prohibited in durable prediction bundles")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise PilotError(f"Refusing to overwrite immutable prediction bundle: {path}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive and path.exists():
        tmp.unlink(missing_ok=True)
        raise PilotError(f"Immutable prediction bundle appeared concurrently: {path}")
    os.replace(tmp, path)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def _safe_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _official_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=OFFICIAL_ROOT,
                          check=True, text=True, capture_output=True).stdout.strip()


def _is_committed_and_clean(path: Path) -> bool:
    """Require a tracked path to match the committed HEAD blob exactly."""
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                             text=True, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT,
                           text=True, capture_output=True).returncode == 0
    return tracked and clean


def _historical_paths(origin: int) -> dict[str, Path]:
    a02 = OFFICIAL_ROOT / f"experiments/runs/optimization_v3_a02_neighbors_v1/E{origin}/observed"
    return {
        "a02_model": a02 / "model.joblib",
        "a02_fit_state": a02 / "fit_state.json",
        "a02_fit_receipt": a02 / "fit_receipt.json",
        "a02_predictions": a02 / "predictions.npz",
        "a02_prediction_manifest": a02 / "predictions.json",
        "pos_predictions": POS_ROOT / f"E{origin}/predictions_E{origin}.npz",
        "pos_prediction_verification": POS_ROOT / f"PREDICTION_VERIFICATION_E{origin}.json",
        "pos_matrix_manifest": POS_ROOT / f"MATRIX_MANIFEST_E{origin}.json",
        "pos_freeze": POS_ROOT / "FREEZE.json",
        "pos_offset_predictions": POS_OFFSET_ROOT / f"E{origin}/predictions_E{origin}.npz",
        "pos_offset_receipt": POS_OFFSET_ROOT / f"receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json",
        "pos_offset_unit": POS_OFFSET_ROOT / f"units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
        "poisson_parent_predictions": POISSON_PARENT_ROOT / f"E{origin}/predictions_E{origin}.npz",
        "train": TRAIN,
    }


def _input_path_label(path: Path) -> str:
    path = path.resolve()
    if path.is_relative_to(REPO_ROOT):
        return f"REPO::{path.relative_to(REPO_ROOT).as_posix()}"
    if path.is_relative_to(OFFICIAL_ROOT):
        return f"OFFICIAL::{path.relative_to(OFFICIAL_ROOT).as_posix()}"
    raise PilotError(f"Frozen input is outside both source repositories: {path}")


def _source_file_hashes() -> dict[str, str]:
    paths = [
        PROTOCOL,
        ROOT_PLAN,
        ROOT_ADJUDICATION,
        INCIDENT,
        AMENDMENT,
        REPO_ROOT / "src/models/a02_position_kernel_v1.py",
        REPO_ROOT / "scripts/cls_a02_position_kernel_v1.py",
        REPO_ROOT / "scripts/verify_cls_a02_position_kernel_v1_prelabel.py",
        REPO_ROOT / "scripts/score_cls_a02_position_kernel_v1.py",
        REPO_ROOT / "scripts/verify_cls_a02_position_kernel_v1_metrics.py",
        REPO_ROOT / "tests/test_cls_a02_position_kernel_v1.py",
        REPO_ROOT / "scripts/cls_pos_poisson_bag2_v1.py",
        REPO_ROOT / "scripts/verify_cls_pos_poisson_bag2_v1.py",
        REPO_ROOT / "src/models/cls_pos_poisson_bag2_v1.py",
        POISSON_PARENT_REPORT / "PROTOCOL.md",
        POISSON_PARENT_FREEZE,
        POISSON_PARENT_ROOT_PASS,
        REPORT_ROOT / "ROOT_PARENT_SOURCE_REVIEW.md",
        OFFICIAL_ROOT / "src/models/classification_a02_neighbors.py",
        OFFICIAL_ROOT / "src/models/optimization_v3_positional_text_pilot_v1.py",
        OFFICIAL_ROOT / "scripts/optimization_v3_positional_text_pilot_v1.py",
        OFFICIAL_ROOT / "src/models/positional_text_features.py",
        OFFICIAL_ROOT / "src/models/positional_bag_offset_v1.py",
        OFFICIAL_ROOT / "scripts/optimization_v3_positional_bag_offset_v1.py",
        OFFICIAL_ROOT / "scripts/verify_optimization_v3_positional_bag_offset_v1.py",
        OFFICIAL_ROOT / "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
        OFFICIAL_ROOT / "reports/optimization_v3/positional_text_diagnostic_v1/SOL_OFFSET_PILOT_AUTHORIZATION.md",
        OFFICIAL_ROOT / "src/models/a02_coefficient_geometry_v1.py",
        OFFICIAL_ROOT / "src/models/classification_models.py",
        OFFICIAL_ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json",
        OFFICIAL_ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/FINAL_HANDOFF.md",
        OFFICIAL_ROOT / "reports/optimization_v3/posoffset_a02_endgame_v1/SOL_FINAL_DECISION.md",
    ]
    return {str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT)
            else f"OFFICIAL::{path.relative_to(OFFICIAL_ROOT).as_posix()}": _sha_file(path)
            for path in paths}


def _input_file_hashes() -> dict[str, str]:
    paths: list[Path] = [
        TRAIN,
        POS_ROOT / "FREEZE.json",
        POS_ROOT / "PREDICTIONS_READY.json",
        POS_OFFSET_ROOT / "FREEZE.json",
        POS_OFFSET_ROOT / "PREDICTIONS_READY.json",
        POS_OFFSET_ROOT / "PRELABEL_VERIFICATION.json",
        POISSON_PARENT_FREEZE,
        POISSON_PARENT_ROOT_PASS,
        POISSON_PARENT_READY,
        POISSON_PARENT_READBACK,
    ]
    for origin in ORIGINS:
        paths.extend(_historical_paths(origin).values())
        paths.append(POS_ROOT / f"receipts/POSITIONAL_TEXT_PILOT_V1_E{origin}_POSITION_LR.json")
        paths.append(POS_ROOT / f"units/POSITIONAL_TEXT_PILOT_V1_E{origin}_POSITION_LR.joblib")
        paths.append(POS_OFFSET_ROOT / f"receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json")
        paths.append(POS_OFFSET_ROOT / f"units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz")
    unique = sorted(set(paths))
    result: dict[str, str] = {}
    for path in unique:
        if not path.is_file():
            raise PilotError(f"Frozen input is missing: {path}")
        label = _input_path_label(path)
        result[label] = _sha_file(path)
    return result


def _normal_text(series: pd.Series) -> list[str]:
    return ["" if pd.isna(value) else str(value) for value in series.tolist()]


def _producer_canonical_ids(raw_values: Any) -> tuple[str, ...]:
    """Reproduce A02's exact NumPy-string IDs, including the ``str_`` type tag."""
    raw = np.asarray(raw_values)
    if raw.ndim != 1 or pd.isna(raw).any():
        raise PilotError("A02 donor record IDs must be a nonnull one-dimensional array")
    producer_values = raw.astype(str)
    return tuple(f"{type(value).__name__}:{value}" for value in producer_values)


def _validate_donor_identity(
    raw_ids: Any,
    raw_cycles: Any,
    expected_ids: Any,
    expected_cycles: Any,
    *,
    origin: int,
) -> None:
    if _producer_canonical_ids(raw_ids) != tuple(expected_ids):
        raise PilotError(f"Original A02 donor IDs differ from official Train row identity for E{origin}")
    cycles = np.asarray(raw_cycles, dtype=np.int64)
    frozen_cycles = np.asarray(expected_cycles, dtype=np.int64)
    if cycles.shape != frozen_cycles.shape or not np.array_equal(cycles, frozen_cycles):
        raise PilotError(f"Original A02 donor cycles differ from official Train for E{origin}")


def _validate_historical_artifacts(origin: int, *, input_hashes: dict[str, str] | None = None) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    paths = _historical_paths(origin)
    fit_receipt = _safe_json(paths["a02_fit_receipt"])
    fit_state = _safe_json(paths["a02_fit_state"])
    a02_manifest = _safe_json(paths["a02_prediction_manifest"])
    pos_verify = _safe_json(paths["pos_prediction_verification"])
    pos_manifest = _safe_json(paths["pos_matrix_manifest"])
    pos_freeze = _safe_json(paths["pos_freeze"])
    offset_freeze = _safe_json(POS_OFFSET_ROOT / "FREEZE.json")
    offset_ready = _safe_json(POS_OFFSET_ROOT / "PREDICTIONS_READY.json")
    offset_prelabel = _safe_json(POS_OFFSET_ROOT / "PRELABEL_VERIFICATION.json")
    parent_freeze = _safe_json(POISSON_PARENT_FREEZE)
    parent_root_pass = _safe_json(POISSON_PARENT_ROOT_PASS)
    parent_ready = _safe_json(POISSON_PARENT_READY)
    parent_readback = _safe_json(POISSON_PARENT_READBACK)

    if fit_receipt.get("fit_id") != f"A02_E{origin}_OBSERVED":
        raise PilotError(f"A02 fit receipt identity is invalid for E{origin}")
    if fit_receipt.get("model_sha256") != _sha_file(paths["a02_model"]):
        raise PilotError(f"A02 saved model hash does not match its fit receipt for E{origin}")
    if a02_manifest.get("npz_sha256") != _sha_file(paths["a02_predictions"]):
        raise PilotError(f"A02 prediction bundle hash does not match its manifest for E{origin}")
    if pos_verify.get("status") != "PASS_PREDICTION_INTEGRITY":
        raise PilotError(f"Original POS prediction verification did not pass for E{origin}")
    if pos_verify.get("prediction_file_sha256") != _sha_file(paths["pos_predictions"]):
        raise PilotError(f"Original POS prediction file hash mismatch for E{origin}")
    if pos_manifest.get("origin") != origin or pos_verify.get("origin") != origin:
        raise PilotError(f"Original POS cache origin mismatch for E{origin}")
    if pos_freeze.get("pilot_id") != "POSITIONAL_TEXT_PILOT_V1":
        raise PilotError("Original POS cache is not bound to its frozen pilot")

    if (offset_freeze.get("card_id") != "POS_BAG_OFFSET_V1"
            or offset_ready.get("card_id") != "POS_BAG_OFFSET_V1"
            or offset_ready.get("freeze_signature") != offset_freeze.get("freeze_signature")
            or offset_ready.get("source_commit") != offset_freeze.get("source_commit")
            or offset_ready.get("query_labels_opened") is not False
            or offset_ready.get("all_four_gamma_receipts_durable") is not True):
        raise PilotError("Frozen original POS offset prediction lineage is incomplete")
    if (offset_prelabel.get("status") != "PASS"
            or offset_prelabel.get("query_labels_read") is not False
            or offset_prelabel.get("preprocessing_fits") != 0
            or offset_prelabel.get("optimizer_refits") != 0
            or offset_prelabel.get("predictions_ready_sha256") != _sha_file(POS_OFFSET_ROOT / "PREDICTIONS_READY.json")):
        raise PilotError("Original POS offset prelabel verification is incomplete")
    if (parent_freeze.get("candidate_id") != "CLS_POS_POISSON_BAG2_V1"
            or parent_root_pass.get("status") != "PASS_ROOT_PREFIT_APPROVED"
            or parent_root_pass.get("freeze_signature") != parent_freeze.get("freeze_signature")
            or parent_ready.get("candidate_id") != "CLS_POS_POISSON_BAG2_V1"
            or parent_ready.get("freeze_signature") != parent_freeze.get("freeze_signature")
            or parent_ready.get("query_labels_opened") is not False
            or parent_ready.get("test_or_sample_access") is not False
            or parent_ready.get("all_four_fit_receipts_durable") is not True):
        raise PilotError("Verified Poisson original-only champion cache lineage is incomplete")
    if (parent_readback.get("status") != "PASS_INDEPENDENT_PREDICTION_READBACK"
            or parent_readback.get("predictions_ready_sha256") != _sha_file(POISSON_PARENT_READY)):
        raise PilotError("Poisson original-only champion cache lacks independent readback")

    # Bind the cached original-only comparator to the exact code frozen when it
    # was created, and its original POS parent to the source verified below.
    for relative in (
        "scripts/cls_pos_poisson_bag2_v1.py",
        "scripts/verify_cls_pos_poisson_bag2_v1.py",
        "src/models/cls_pos_poisson_bag2_v1.py",
    ):
        expected = parent_freeze.get("audit_source_hashes", {}).get(relative)
        if not expected or expected != _sha_file(REPO_ROOT / relative):
            raise PilotError(f"Poisson parent source hash does not match its prefit freeze: {relative}")
    for relative in (
        "scripts/optimization_v3_positional_bag_offset_v1.py",
        "scripts/verify_optimization_v3_positional_bag_offset_v1.py",
        "src/models/positional_bag_offset_v1.py",
        "reports/optimization_v3/positional_bag_offset_v1/RECIPE.json",
    ):
        expected = offset_freeze.get("source_hashes", {}).get(relative)
        parent_expected = parent_freeze.get("external_source_hashes", {}).get(relative)
        if (not expected or expected != _sha_file(OFFICIAL_ROOT / relative)
                or parent_expected != expected):
            raise PilotError(f"Original POS offset source hash does not match its freeze: {relative}")

    # Ground-truth members stay closed; label_champion below is a prediction.
    a02 = load_npz_allowlisted(paths["a02_predictions"], allowed_members=A02_MEMBERS)
    pos = load_npz_allowlisted(
        paths["pos_predictions"], allowed_members=POS_MEMBERS,
        expected_sha256=pos_verify["prediction_file_sha256"], allow_closed_extras=True,
    )
    offset_key = f"E{origin}"
    offset_output = offset_ready.get("predictions", {}).get(offset_key, {})
    offset_readback_hash = offset_prelabel.get("prediction_output_sha256s", {}).get(offset_key)
    if (offset_output.get("path") != f"experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/E{origin}/predictions_E{origin}.npz"
            or offset_output.get("sha256") != offset_readback_hash
            or offset_output.get("sha256") != _sha_file(paths["pos_offset_predictions"])
            or int(offset_output.get("rows", -1)) != int(len(a02["score_rows"]))
            or int(offset_output.get("present_rows", -1)) != int(np.sum(pos["present_mask"]))
            or int(offset_output.get("fallback_rows", -1)) != int(np.sum(~pos["present_mask"]))):
        raise PilotError(f"Original POS offset output hash is not bound to its frozen/readback receipt for E{origin}")
    offset_receipt = _safe_json(paths["pos_offset_receipt"])
    if (offset_receipt.get("fit_id") != f"POS_BAG_OFFSET_V1_E{origin}_REAL"
            or offset_receipt.get("origin") != origin
            or offset_receipt.get("card_id") != "POS_BAG_OFFSET_V1"
            or offset_receipt.get("freeze_signature") != offset_freeze.get("freeze_signature")
            or offset_receipt.get("artifact_sha256") != _sha_file(paths["pos_offset_unit"])):
        raise PilotError(f"Original POS offset gamma receipt is invalid for E{origin}")
    offset = load_npz_allowlisted(
        paths["pos_offset_predictions"], allowed_members=POS_OFFSET_MEMBERS,
        expected_sha256=offset_output["sha256"], allow_closed_extras=True,
    )

    parent_key = f"E{origin}"
    parent_output = parent_ready.get("predictions", {}).get(parent_key, {})
    parent_readback_output = parent_readback.get("predictions", {}).get(parent_key, {})
    if (parent_output.get("path") != f"experiments/CLS_POS_POISSON_BAG2_V1/E{origin}/predictions_E{origin}.npz"
            or parent_output.get("sha256") != parent_readback_output.get("bundle_sha256")
            or parent_output.get("sha256") != _sha_file(paths["poisson_parent_predictions"])
            or parent_readback_output.get("has_label_values") is not False
            or int(parent_output.get("rows", -1)) != int(len(a02["score_rows"]))
            or int(parent_output.get("missing_rows", -1)) != int(np.sum(a02["missing_mask"]))):
        raise PilotError(f"Poisson original-only cache hash/readback is invalid for E{origin}")
    parent = load_npz_allowlisted(
        paths["poisson_parent_predictions"], allowed_members=POISSON_PARENT_MEMBERS,
        expected_sha256=parent_output["sha256"], allow_closed_extras=True,
    )

    if not np.array_equal(a02["score_rows"], pos["score_rows"]):
        raise PilotError(f"A02 and POS query row positions differ for E{origin}")
    if not np.array_equal(a02["score_record_ids"], pos["score_record_ids"]):
        raise PilotError(f"A02 and POS query record IDs differ for E{origin}")
    if not np.array_equal(a02["score_cycles"], pos["score_cycles"]):
        raise PilotError(f"A02 and POS query cycles differ for E{origin}")
    if not np.array_equal(pos["present_mask"], ~a02["missing_mask"]):
        raise PilotError(f"Original POS availability route differs for E{origin}")
    if len(a02["score_rows"]) != int(pos_verify["score_rows"]):
        raise PilotError(f"Historical prediction row count mismatch for E{origin}")
    if (not np.array_equal(offset["score_rows"], a02["score_rows"])
            or not np.array_equal(offset["score_record_ids"].astype(str), a02["score_record_ids"].astype(str))
            or not np.array_equal(offset["score_cycles"], a02["score_cycles"])
            or not np.array_equal(offset["present_mask"], pos["present_mask"])):
        raise PilotError(f"Original POS offset rows/IDs/cycles/availability differ for E{origin}")
    if (int(parent["origin"].reshape(-1)[0]) != origin
            or not np.array_equal(parent["row_positions"], a02["score_rows"])
            or not np.array_equal(parent["record_ids"].astype(str), a02["score_record_ids"].astype(str))
            or not np.array_equal(parent["cycles"], a02["score_cycles"])
            or not np.array_equal(parent["present_mask"], pos["present_mask"])
            or str(parent["input_freeze_signature"].reshape(-1)[0]) != parent_freeze["freeze_signature"]
            or str(parent["matrix_manifest_sha256"].reshape(-1)[0]) != parent_output.get("matrix_manifest_sha256")
            or parent_readback_output.get("ids_sha256") != _canonical_sha(parent["record_ids"].astype(str).tolist())):
        raise PilotError(f"Poisson original-only cache rows/IDs/cycles/availability differ for E{origin}")
    if not np.array_equal(parent["p_a02"], a02["prediction"]):
        raise PilotError(f"Poisson full champion does not use the exact original A02 probabilities for E{origin}")
    present = pos["present_mask"]
    if not np.array_equal(parent["p_pos_original"][present], offset["prob_offset_real"][present]):
        raise PilotError(f"Poisson original POS branch differs from frozen prob_offset_real for E{origin}")
    champion_score, champion_class = assemble_full_champion(
        parent["p_pos_original"], a02["prediction"], a02["missing_mask"],
    )
    if not np.array_equal(champion_score.view(np.uint64), parent["p_champion"].view(np.uint64)):
        raise PilotError(f"Poisson cached p_champion does not exactly reconstruct from the original POS/A02 route for E{origin}")
    cached_champion_class = np.asarray(parent["label_champion"])
    if (cached_champion_class.shape != champion_class.shape
            or not np.isin(cached_champion_class, [0, 1]).all()
            or not np.array_equal(champion_class, cached_champion_class.astype(np.int8))):
        raise PilotError(f"Poisson cached champion decisions do not reconstruct at the frozen thresholds for E{origin}")

    model = joblib.load(paths["a02_model"])
    if model.origin_cycle != origin or model.audit_state() != fit_state:
        raise PilotError(f"Loaded original A02 model does not match its saved fit state for E{origin}")
    if len(model.bank_ids) != int(fit_state["bank_rows"]):
        raise PilotError(f"A02 donor-bank size mismatch for E{origin}")
    if input_hashes is not None:
        global_inputs = (
            POS_ROOT / "FREEZE.json",
            POS_ROOT / "PREDICTIONS_READY.json",
            POS_OFFSET_ROOT / "FREEZE.json",
            POS_OFFSET_ROOT / "PREDICTIONS_READY.json",
            POS_OFFSET_ROOT / "PRELABEL_VERIFICATION.json",
            POISSON_PARENT_FREEZE,
            POISSON_PARENT_ROOT_PASS,
            POISSON_PARENT_READY,
            POISSON_PARENT_READBACK,
        )
        for path in (*paths.values(), *global_inputs):
            label = _input_path_label(path)
            if input_hashes.get(label) != _sha_file(path):
                raise PilotError(f"Frozen input hash changed after freeze: {label}")
    lineage = {
        "fit_receipt": fit_receipt,
        "fit_state": fit_state,
        "a02_manifest": a02_manifest,
        "pos_verify": pos_verify,
        "pos_unused_member_names": list(pos.unused_member_names),
        "pos_manifest": pos_manifest,
        "pos_freeze": pos_freeze,
        "pos_offset_ready": offset_ready,
        "pos_offset_prelabel": offset_prelabel,
        "pos_offset_receipt": offset_receipt,
        "pos_offset_unused_member_names": list(offset.unused_member_names),
        "poisson_parent_freeze_signature": parent_freeze["freeze_signature"],
        "poisson_parent_readback_status": parent_readback["status"],
        "poisson_parent_prediction_sha256": parent_output["sha256"],
        "poisson_parent_unused_member_names": list(parent.unused_member_names),
        "p_pos_present_exact_with_prob_offset_real": True,
        "p_a02_bit_exact_with_original_a02": True,
        "p_champion_bit_exact_reconstruction": True,
        "champion_decisions_exact_reconstruction": True,
    }
    return model, a02, pos, parent, lineage


def _load_target_free_frame(required_rows: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "signal_stream"],
                        dtype={"record_id": "string"})
    rows = np.asarray(required_rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(frame)):
        raise PilotError("Official Train row positions are invalid")
    return frame


def _load_origin_inputs(origin: int, *, input_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    model, a02, pos, parent, lineage = _validate_historical_artifacts(origin, input_hashes=input_hashes)
    query_rows = np.asarray(a02["score_rows"], dtype=np.int64)
    bank_rows = np.asarray(model.bank_row_positions, dtype=np.int64)
    frame = _load_target_free_frame(np.concatenate((query_rows, bank_rows)))
    query_raw = frame.iloc[query_rows]
    bank_raw = frame.iloc[bank_rows]
    query_ids = query_raw["record_id"].astype("string").fillna("").astype(str).to_numpy()
    query_cycles = query_raw["cycle_code"].to_numpy(dtype=np.int64)
    if not np.array_equal(query_ids.astype(a02["score_record_ids"].dtype), a02["score_record_ids"]):
        raise PilotError(f"Official Train query record IDs differ from frozen caches for E{origin}")
    if not np.array_equal(query_cycles, a02["score_cycles"]):
        raise PilotError(f"Official Train query cycles differ from frozen caches for E{origin}")
    _validate_donor_identity(
        bank_raw["record_id"].to_numpy(),
        bank_raw["cycle_code"].to_numpy(dtype=np.int64),
        model.bank_ids,
        model.bank_cycles,
        origin=origin,
    )
    if np.any(model.bank_cycles > origin) or np.any(query_cycles <= origin):
        raise PilotError(f"Original A02 bank/query temporal boundary is invalid for E{origin}")
    query_docs, query_missing = model.text_space.transform(_normal_text(query_raw["signal_stream"]))
    bank_texts = _normal_text(bank_raw["signal_stream"])
    if not np.array_equal(query_missing, a02["missing_mask"]):
        raise PilotError(f"Official Train query missing-stream mask differs from A02 cache for E{origin}")
    if np.any(pd.isna(bank_raw["signal_stream"]).to_numpy()):
        raise PilotError(f"Original A02 donor bank unexpectedly contains missing text for E{origin}")
    original_reconstructed = np.clip(a02["baseline_prediction"] + a02["correction"], 0.0, 1.0)
    if not np.allclose(original_reconstructed, a02["prediction"], rtol=0.0, atol=1e-12):
        raise PilotError(f"Cached A02 probabilities do not reconstruct from Sparse25 plus correction for E{origin}")
    m0 = np.sum(a02["neighbor_similarities"] ** 2, axis=1, dtype=np.float64)
    if not np.array_equal(m0[a02["missing_mask"]], np.zeros(int(a02["missing_mask"].sum()))):
        raise PilotError(f"Original A02 missing rows have nonzero mass for E{origin}")
    bank_set = set(model.bank_row_positions.tolist())
    for row in a02["neighbor_source_rows"]:
        for donor in row[row >= 0].tolist():
            if int(donor) not in bank_set:
                raise PilotError(f"Original A02 cached donor falls outside the full bank for E{origin}")
    return {
        "origin": origin,
        "model": model,
        "a02": a02,
        "pos": pos,
        "poisson_parent": parent,
        "lineage": lineage,
        "train_frame": frame,
        "query_raw": query_raw,
        "bank_texts": bank_texts,
        "query_texts": _normal_text(query_raw["signal_stream"]),
        "query_bag": query_docs,
        "query_ids": query_ids,
        "query_rows": query_rows,
        "query_cycles": query_cycles,
        "p_pos": np.asarray(parent["p_pos_original"], dtype=np.float64),
        "missing": np.asarray(a02["missing_mask"], dtype=bool),
    }


def _candidate_arrays(origin: int, arm: str, inputs: dict[str, Any], fit_meta: dict[str, Any], query_meta: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model = inputs["model"]
    a02 = inputs["a02"]
    out = construct_position_kernel(
        text_space=model.text_space,
        query_bag=inputs["query_bag"],
        bank_bag=model.bank_matrix,
        query_position=query_meta["matrix"],
        bank_position=fit_meta["matrix"],
        bank_ids=model.bank_ids,
        bank_row_positions=model.bank_row_positions,
        bank_cycles=model.bank_cycles,
        bank_residuals=model.bank_residuals,
        query_ids=inputs["query_ids"],
        query_row_positions=inputs["query_rows"],
        query_cycles=inputs["query_cycles"],
        baseline_probabilities=a02["baseline_prediction"],
        original_a02_probabilities=a02["prediction"],
        original_a02_neighbor_rows=a02["neighbor_source_rows"],
        original_a02_neighbor_similarities=a02["neighbor_similarities"],
        missing_mask=inputs["missing"],
        arm=arm,
        block_rows=16,
    )
    champion_score, champion_class = assemble_full_champion(inputs["p_pos"], a02["prediction"], inputs["missing"])
    candidate_score, candidate_class = assemble_full_champion(inputs["p_pos"], out["candidate_a02_probability"], inputs["missing"])
    arrays = {
        "score_rows": inputs["query_rows"].astype(np.int64, copy=True),
        "score_record_ids": a02["score_record_ids"].astype(str),
        "score_cycles": inputs["query_cycles"].astype(np.int16, copy=True),
        "missing_mask": inputs["missing"].astype(bool, copy=True),
        "entry_friction": a02["entry_friction"].astype(np.int8, copy=True),
        "text_load_group": a02["text_load_group"].astype(np.int8, copy=True),
        "p_sparse25": a02["baseline_prediction"].astype(np.float64, copy=True),
        "p_a02_original": a02["prediction"].astype(np.float64, copy=True),
        "p_pos_original": inputs["p_pos"].astype(np.float64, copy=True),
        "p_a02_candidate": out["candidate_a02_probability"].astype(np.float64, copy=True),
        "champion_score": champion_score.astype(np.float64, copy=True),
        "candidate_score": candidate_score.astype(np.float64, copy=True),
        "champion_class": champion_class.astype(np.int8, copy=True),
        "candidate_class": candidate_class.astype(np.int8, copy=True),
        "m0": out["m0"].astype(np.float64, copy=True),
        "neighbor_count": out["neighbor_count"].astype(np.int16, copy=True),
        "neighbor_source_rows": out["neighbor_source_rows"].astype(np.int64, copy=True),
        "neighbor_ids": out["neighbor_ids"].astype(str),
        "neighbor_kernel_similarity": out["neighbor_kernel_similarity"].astype(np.float64, copy=True),
        "neighbor_weights": out["neighbor_weights"].astype(np.float64, copy=True),
        "position_kernel_positive_mass": out["position_kernel_positive_mass"].astype(np.float64, copy=True),
        "correction_raw": out["correction_raw"].astype(np.float64, copy=True),
        "correction": out["correction"].astype(np.float64, copy=True),
        "mass_error": out["mass_error"].astype(np.float64, copy=True),
        "neighbor_overlap_fraction": out["neighbor_overlap_fraction"].astype(np.float64, copy=True),
    }
    return arrays, {"kernel": out["diagnostics"], "fit_position": fit_meta["metadata"],
                    "query_position": query_meta["metadata"]}


def _append_event(event: dict[str, Any], *, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_events(path: Path = LEDGER) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotError("Lane budget ledger is missing")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resource_snapshot() -> dict[str, int | bool]:
    import psutil
    available = int(psutil.virtual_memory().available)
    rss = int(psutil.Process().memory_info().rss)
    return {
        "available_bytes": available,
        "process_rss_bytes": rss,
        "estimated_additional_bytes": ESTIMATED_ADDITIONAL,
        "projected_available_bytes": available - ESTIMATED_ADDITIONAL,
        "initial_ram_pass": available >= INITIAL_RAM_MIN,
        "projected_reserve_pass": available - ESTIMATED_ADDITIONAL >= PROJECTED_RESERVE,
        "max_threads": MAX_THREADS,
    }


@contextmanager
def _advisory_lock(path: Path, *, wait: bool) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"\0")
        stream.flush()
    unlock = None
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                    break
                except OSError as exc:
                    if not wait:
                        raise PilotError(f"Shared heavy-job lock is held: {path}") from exc
                    print(f"Waiting for KASDAD shared lock: {path}", flush=True)
                    time.sleep(5)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    break
                except OSError as exc:
                    if not wait:
                        raise PilotError(f"Shared heavy-job lock is held: {path}") from exc
                    print(f"Waiting for KASDAD shared lock: {path}", flush=True)
                    time.sleep(5)
        yield
    finally:
        if unlock is not None:
            stream.seek(0)
            unlock()
        stream.close()


@contextmanager
def _unit_locks(unit_id: str) -> Any:
    shared = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
    local = RUN_ROOT / "LANE.lock"
    with _advisory_lock(shared, wait=True):
        with _advisory_lock(local, wait=False):
            yield


def _freeze_payload() -> dict[str, Any]:
    if _official_commit() != EXPECTED_OFFICIAL_COMMIT:
        raise PilotError(f"Official source checkout changed: expected {EXPECTED_OFFICIAL_COMMIT}")
    protocol = _safe_json(PROTOCOL)
    if protocol.get("pilot_id") != PILOT_ID or protocol.get("authorized_predictor_constructions") != 4:
        raise PilotError("Frozen protocol identity or cap differs")
    file_hashes = _source_file_hashes()
    input_hashes = _input_file_hashes()
    views: dict[str, Any] = {}
    loaded: dict[int, dict[str, Any]] = {}
    for origin in ORIGINS:
        inputs = _load_origin_inputs(origin, input_hashes=input_hashes)
        loaded[origin] = inputs
        views[f"E{origin}"] = {
            "query_rows": int(len(inputs["query_rows"])),
            "missing_rows": int(inputs["missing"].sum()),
            "bank_rows": int(len(inputs["model"].bank_ids)),
            "bank_cycle_min": int(np.min(inputs["model"].bank_cycles)),
            "bank_cycle_max": int(np.max(inputs["model"].bank_cycles)),
            "query_cycle_min": int(np.min(inputs["query_cycles"])),
            "query_cycle_max": int(np.max(inputs["query_cycles"])),
            "rows_sha256": _canonical_sha(inputs["query_rows"].astype(int).tolist()),
            "record_ids_sha256": _canonical_sha(inputs["a02"]["score_record_ids"].astype(str).tolist()),
            "pos_archive_unused_member_names": inputs["lineage"]["pos_unused_member_names"],
            "pos_offset_archive_unused_member_names": inputs["lineage"]["pos_offset_unused_member_names"],
            "poisson_parent_archive_unused_member_names": inputs["lineage"]["poisson_parent_unused_member_names"],
            "champion_parent": {
                "source": "CLS_POS_POISSON_BAG2_V1 original-only p_pos_original and p_champion",
                "poisson_freeze_signature": inputs["lineage"]["poisson_parent_freeze_signature"],
                "poisson_prediction_sha256": inputs["lineage"]["poisson_parent_prediction_sha256"],
                "readback_status": inputs["lineage"]["poisson_parent_readback_status"],
                "p_pos_matches_prob_offset_real_present": inputs["lineage"]["p_pos_present_exact_with_prob_offset_real"],
                "p_a02_matches_original_a02": inputs["lineage"]["p_a02_bit_exact_with_original_a02"],
                "p_champion_reconstructs_bit_exactly": inputs["lineage"]["p_champion_bit_exact_reconstruction"],
                "champion_decisions_reconstruct_exactly": inputs["lineage"]["champion_decisions_exact_reconstruction"],
            },
        }
    e67_rows = loaded[67]
    e75_rows = loaded[75]
    pool_ids = set(e67_rows["a02"]["score_record_ids"][np.isin(e67_rows["query_cycles"], np.arange(68, 76))].astype(str))
    pool_ids.update(e75_rows["a02"]["score_record_ids"][np.isin(e75_rows["query_cycles"], np.arange(76, 80))].astype(str))
    if len(pool_ids) != 20903:
        raise PilotError(f"Unique primary pool identity check failed: n={len(pool_ids)}")
    return {
        "pilot_id": PILOT_ID,
        "created_at_utc": _utc(),
        "source_commit": _git("rev-parse", "HEAD"),
        "official_source_commit": _official_commit(),
        "protocol_sha256": _sha_file(PROTOCOL),
        "root_plan_sha256": _sha_file(ROOT_PLAN),
        "source_hashes": file_hashes,
        "input_hashes": input_hashes,
        "frozen_views": views,
        "unique_primary_rows": len(pool_ids),
        "units": list(FIT_IDS),
        "estimator_fits": 0,
        "predictor_constructions": 4,
        "query_labels_opened_by_this_freeze_step": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_accessed": False,
        "original_prelabel_boundary_compliant": False,
        "scientific_qualification": False,
        "deviation_incident": "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1/INCIDENT_01.json",
        "resource_limits": {"threads": MAX_THREADS, "initial_free_ram_min_bytes": INITIAL_RAM_MIN,
                            "projected_reserve_bytes": PROJECTED_RESERVE},
    }


def prepare_freeze() -> dict[str, Any]:
    if FREEZE.exists():
        existing = _safe_json(FREEZE)
        if existing.get("pilot_id") != PILOT_ID:
            raise PilotError("Existing freeze belongs to another pilot")
        if existing.get("source_hashes") != _source_file_hashes() or existing.get("input_hashes") != _input_file_hashes():
            raise PilotError("Existing freeze no longer matches current source or inputs")
        return existing
    if _git("status", "--porcelain"):
        raise PilotError("Freeze creation requires all implementation source to be committed")
    value = _freeze_payload()
    _atomic_json(FREEZE, value, exclusive=True)
    return value


def initialize_budget() -> None:
    if not FREEZE.is_file():
        raise PilotError("Cannot reserve units before the frozen plan exists")
    freeze = _safe_json(FREEZE)
    if freeze.get("source_commit") not in _git("rev-list", "HEAD").splitlines():
        raise PilotError("Frozen source commit is not an ancestor of HEAD")
    if not _is_committed_and_clean(FREEZE):
        raise PilotError("Freeze must be committed and unchanged before reserving the four units")
    if LEDGER.exists():
        events = _ledger_events()
        reserved = [event.get("fit_id") for event in events if event.get("event") == "PREDICTOR_RESERVED"]
        caps = [event for event in events if event.get("event") == "CAP_SET"]
        if (len(caps) == 1 and caps[0].get("pilot_id") == PILOT_ID and caps[0].get("cap") == 4
                and caps[0].get("units") == list(FIT_IDS)
                and sorted(reserved) == sorted(FIT_IDS) and len(reserved) == len(FIT_IDS)):
            return
        raise PilotError("Existing lane ledger is partial or differs; it will not be reset")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("x", encoding="utf-8", newline="\n") as stream:
        cap_event = {"event": "CAP_SET", "pilot_id": PILOT_ID, "cap": 4,
                     "units": list(FIT_IDS), "source_commit": freeze["source_commit"], "created_at_utc": _utc()}
        stream.write(json.dumps(cap_event, sort_keys=True, separators=(",", ":")) + "\n")
        for fit_id in FIT_IDS:
            event = {"event": "PREDICTOR_RESERVED", "fit_id": fit_id, "pilot_id": PILOT_ID,
                     "reservation_id": f"{PILOT_ID}:{fit_id}", "source_commit": freeze["source_commit"],
                     "created_at_utc": _utc()}
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _output_paths(unit_id: str) -> tuple[Path, Path]:
    base = RUN_ROOT / unit_id
    return base / "predictions.npz", base / "receipt.json"


def _read_output(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if names & FORBIDDEN_OUTPUT_MEMBERS:
            raise PilotError(f"Target-bearing output member found: {sorted(names & FORBIDDEN_OUTPUT_MEMBERS)}")
        if names != set(OUTPUT_MEMBERS):
            raise PilotError(f"Prediction output schema mismatch: {sorted(names ^ set(OUTPUT_MEMBERS))}")
        return {name: archive[name].copy() for name in OUTPUT_MEMBERS}


def persist_or_resume_unit(
    unit_id: str,
    build: Any,
    *,
    artifact_path: Path,
    receipt_path: Path,
    ledger_path: Path,
    lineage: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    """Persist one constructed bundle or recover an already durable bundle.

    All filesystem paths are explicit. Synthetic lifecycle tests therefore
    cannot write to the production namespace by default.
    """
    events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reserved = any(e.get("event") == "PREDICTOR_RESERVED" and e.get("fit_id") == unit_id for e in events)
    started = any(e.get("event") == "PREDICTOR_STARTED" and e.get("fit_id") == unit_id for e in events)
    saved = any(e.get("event") == "PREDICTOR_SAVED" and e.get("fit_id") == unit_id for e in events)
    if not reserved:
        raise PilotError(f"Unit has no explicit reservation: {unit_id}")
    if artifact_path.exists() != receipt_path.exists():
        raise PilotError(f"Partial durable state for {unit_id}; no retry or overwrite is allowed")
    if artifact_path.exists():
        receipt = _safe_json(receipt_path)
        if receipt.get("fit_id") != unit_id or receipt.get("artifact_sha256") != _sha_file(artifact_path):
            raise PilotError(f"Durable state receipt mismatch for {unit_id}")
        if receipt.get("lineage") != lineage:
            raise PilotError(f"Durable state lineage mismatch for {unit_id}")
        arrays = _read_output(artifact_path)
        if not saved:
            _append_event({"event": "PREDICTOR_SAVED", "fit_id": unit_id, "pilot_id": PILOT_ID,
                           "artifact_sha256": _sha_file(artifact_path),
                           "receipt_sha256": _sha_file(receipt_path),
                           "resumed_existing_durable_output": True, "saved_at_utc": _utc()}, path=ledger_path)
        return {"status": "RESUMED_DURABLE", "fit_id": unit_id,
                "artifact_sha256": _sha_file(artifact_path), "arrays": arrays,
                "receipt_sha256": _sha_file(receipt_path)}
    if started:
        raise PilotError(f"A start already exists without durable output for {unit_id}; never retry a consumed unit")
    _append_event({"event": "PREDICTOR_STARTED", "fit_id": unit_id, "pilot_id": PILOT_ID,
                   "resource": resource, "started_at_utc": _utc()}, path=ledger_path)
    started_at = time.perf_counter()
    arrays, diagnostics = build()
    _atomic_npz(artifact_path, arrays, exclusive=True)
    receipt = {
        "pilot_id": PILOT_ID,
        "fit_id": unit_id,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha_file(artifact_path),
        "output_members": sorted(arrays),
        "query_target_included": False,
        "predictor_constructions": 1,
        "estimator_fits": 0,
        "preprocessing_fits": 0,
        "lineage": lineage,
        "diagnostics": diagnostics,
        "resource_before": resource,
        "elapsed_seconds": time.perf_counter() - started_at,
        "created_at_utc": _utc(),
    }
    _atomic_json(receipt_path, receipt, exclusive=True)
    _append_event({"event": "PREDICTOR_SAVED", "fit_id": unit_id, "pilot_id": PILOT_ID,
                   "artifact_sha256": receipt["artifact_sha256"],
                   "receipt_sha256": _sha_file(receipt_path), "saved_at_utc": _utc()}, path=ledger_path)
    return {"status": "SAVED", "fit_id": unit_id,
            "artifact_sha256": receipt["artifact_sha256"],
            "receipt_sha256": _sha_file(receipt_path), "diagnostics": diagnostics}


def run_unit(origin: int, arm: str) -> dict[str, Any]:
    origin = int(origin)
    arm = arm.upper()
    unit_id = f"E{origin}_{arm}"
    if unit_id not in FIT_IDS:
        raise PilotError(f"Unauthorized unit ID: {unit_id}")
    if not FREEZE.is_file() or not LEDGER.is_file():
        raise PilotError("Frozen source and four-unit reservation must be durable before construction")
    if not _is_committed_and_clean(FREEZE):
        raise PilotError("Committed freeze must remain unchanged before construction")
    freeze = _safe_json(FREEZE)
    if freeze.get("pilot_id") != PILOT_ID or freeze.get("source_commit") not in _git("rev-list", "HEAD").splitlines():
        raise PilotError("Current source does not match the committed pilot freeze")
    if freeze.get("source_hashes") != _source_file_hashes() or freeze.get("input_hashes") != _input_file_hashes():
        raise PilotError("Source or frozen inputs changed after freeze")
    artifact, receipt_path = _output_paths(unit_id)
    events = _ledger_events()
    reserved = {e.get("fit_id") for e in events if e.get("event") == "PREDICTOR_RESERVED"}
    if unit_id not in reserved:
        raise PilotError(f"Unit is not reserved: {unit_id}")
    with _unit_locks(unit_id):
        resource = _resource_snapshot()
        if not resource["initial_ram_pass"] or not resource["projected_reserve_pass"]:
            raise PilotError(f"Resource gate failed before {unit_id}: {resource}")
        def build() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            with threadpool_limits(limits=MAX_THREADS):
                inputs = _load_origin_inputs(origin, input_hashes=freeze["input_hashes"])
                from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block
                from src.models.classification_a02_neighbors import _documents
                bank_docs, bank_missing = _documents(inputs["bank_texts"])
                query_docs, query_missing = _documents(inputs["query_texts"])
                if np.any(bank_missing) or not np.array_equal(query_missing, inputs["missing"]):
                    raise PilotError(f"Position transform text normalization changed at {unit_id}")
                shuffled = arm == "SHUFFLED"
                fit_matrix, fit_meta = build_position_block(
                    bank_docs, inputs["model"].text_space.vectorizer,
                    shuffled=shuffled, seed=SHUFFLE_SEED,
                )
                query_matrix, query_meta = build_position_block(
                    query_docs, inputs["model"].text_space.vectorizer,
                    shuffled=shuffled, seed=SHUFFLE_SEED,
                )
                if shuffled and (fit_meta["bag_preserved_rows"] != len(bank_docs)
                                 or query_meta["bag_preserved_rows"] != len(query_docs)
                                 or fit_meta["segment_sizes_preserved_rows"] != len(bank_docs)
                                 or query_meta["segment_sizes_preserved_rows"] != len(query_docs)):
                    raise PilotError(f"Within-document shuffle failed bag/segment invariants at {unit_id}")
                arrays, diagnostics = _candidate_arrays(
                    origin, arm, inputs,
                    {"matrix": fit_matrix, "metadata": fit_meta},
                    {"matrix": query_matrix, "metadata": query_meta},
                )
                del fit_matrix, query_matrix
                return arrays, diagnostics

        lineage = {"source_commit": freeze["source_commit"],
                   "official_source_commit": freeze["official_source_commit"],
                   "input_hashes": freeze["input_hashes"], "source_hashes": freeze["source_hashes"],
                   "origin": origin, "arm": arm}
        result = persist_or_resume_unit(unit_id, build, artifact_path=artifact,
                                        receipt_path=receipt_path, ledger_path=LEDGER,
                                        lineage=lineage, resource=resource)
        # A resumed unit needs its arrays for the caller to inspect the durable
        # state, but the command-line runner must never dump full predictions.
        result.pop("arrays", None)
        return result


def run_all() -> list[dict[str, Any]]:
    return [run_unit(origin, arm) for origin in ORIGINS for arm in ARMS]


def synthetic_lifecycle() -> dict[str, Any]:
    """Exercise feature-to-score persistence and resume using synthetic arrays only."""
    width = 40
    bank_bag = sparse.eye(width, format="csr", dtype=np.float64)
    bank_position = sparse.eye(width, format="csr", dtype=np.float64)
    query_bag = sparse.csr_matrix(
        (np.asarray([1.0, 1.0]), (np.asarray([0, 1]), np.asarray([35, 2]))),
        shape=(4, width),
    )
    query_position = query_bag.copy()
    vectorizer = SimpleNamespace(vocabulary_={f"t{i}": i for i in range(width)})
    text_space = SimpleNamespace(origin_cycle=55, vectorizer=vectorizer)
    bank_rows = np.arange(1000, 1040, dtype=np.int64)
    bank_residuals = np.zeros(width, dtype=np.float64)
    bank_residuals[35] = 0.10
    bank_residuals[2] = -0.20
    original_rows = np.full((4, 32), -1, dtype=np.int64)
    original_sims = np.zeros((4, 32), dtype=np.float64)
    original_rows[0, 0], original_sims[0, 0] = 1000, 0.8
    original_rows[1, 0], original_sims[1, 0] = 1001, 0.6
    missing = np.asarray([False, False, False, True])
    baseline = np.asarray([0.40, 0.60, 0.30, 0.70], dtype=np.float64)
    original = np.asarray([0.43, 0.58, 0.30, 0.70], dtype=np.float64)
    query_ids = np.asarray(["query-0", "query-1", "query-2", "query-3"])
    query_rows = np.asarray([2000, 2001, 2002, 2003], dtype=np.int64)
    query_cycles = np.asarray([56, 57, 58, 59], dtype=np.int16)
    core = construct_position_kernel(
        text_space=text_space, query_bag=query_bag, bank_bag=bank_bag,
        query_position=query_position, bank_position=bank_position,
        bank_ids=[f"str:bank-{i}" for i in range(width)], bank_row_positions=bank_rows,
        bank_cycles=np.full(width, 55, dtype=np.int16), bank_residuals=bank_residuals,
        query_ids=query_ids, query_row_positions=query_rows, query_cycles=query_cycles,
        baseline_probabilities=baseline, original_a02_probabilities=original,
        original_a02_neighbor_rows=original_rows,
        original_a02_neighbor_similarities=original_sims, missing_mask=missing,
        arm="REAL", block_rows=2,
    )
    p_pos = np.asarray([0.70, 0.20, 0.40, 0.10], dtype=np.float64)
    champion_score, champion_class = assemble_full_champion(p_pos, original, missing)
    candidate_score, candidate_class = assemble_full_champion(p_pos, core["candidate_a02_probability"], missing)
    arrays = {
        "score_rows": query_rows.astype(np.int64), "score_record_ids": query_ids.astype("<U16"),
        "score_cycles": query_cycles, "missing_mask": missing,
        "entry_friction": np.asarray([0, 1, 0, 1], dtype=np.int8),
        "text_load_group": np.asarray([0, 1, 2, 3], dtype=np.int8),
        "p_sparse25": baseline, "p_a02_original": original, "p_pos_original": p_pos,
        "p_a02_candidate": core["candidate_a02_probability"],
        "champion_score": champion_score, "candidate_score": candidate_score,
        "champion_class": champion_class, "candidate_class": candidate_class,
        "m0": core["m0"], "neighbor_count": core["neighbor_count"],
        "neighbor_source_rows": core["neighbor_source_rows"], "neighbor_ids": core["neighbor_ids"],
        "neighbor_kernel_similarity": core["neighbor_kernel_similarity"],
        "neighbor_weights": core["neighbor_weights"],
        "position_kernel_positive_mass": core["position_kernel_positive_mass"],
        "correction_raw": core["correction_raw"], "correction": core["correction"],
        "mass_error": core["mass_error"], "neighbor_overlap_fraction": core["neighbor_overlap_fraction"],
    }
    with tempfile.TemporaryDirectory(prefix="cls_a02_position_kernel_synthetic_") as temporary:
        base = Path(temporary)
        ledger = base / "ledger.jsonl"
        artifact = base / "unit" / "predictions.npz"
        receipt = base / "unit" / "receipt.json"
        ledger.write_text(
            json.dumps({"event": "CAP_SET", "cap": 1}) + "\n"
            + json.dumps({"event": "PREDICTOR_RESERVED", "fit_id": "SYNTHETIC"}) + "\n",
            encoding="utf-8",
        )
        build_calls = {"n": 0}

        def build():
            build_calls["n"] += 1
            return arrays, {"synthetic": True, "kernel": core["diagnostics"]}

        lineage = {"synthetic": True, "source_commit": "synthetic"}
        first = persist_or_resume_unit(
            "SYNTHETIC", build, artifact_path=artifact, receipt_path=receipt,
            ledger_path=ledger, lineage=lineage, resource={"synthetic": True},
        )
        if first.get("status") != "SAVED":
            raise PilotError("Synthetic lifecycle did not save its first output")
        events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        ledger.write_text("\n".join(json.dumps(e) for e in events if e.get("event") != "PREDICTOR_SAVED") + "\n",
                          encoding="utf-8")

        def must_not_build():
            raise AssertionError("resume attempted to rebuild a durable predictor")

        resumed = persist_or_resume_unit(
            "SYNTHETIC", must_not_build, artifact_path=artifact, receipt_path=receipt,
            ledger_path=ledger, lineage=lineage, resource={"synthetic": True},
        )
        if resumed.get("status") != "RESUMED_DURABLE" or build_calls["n"] != 1:
            raise PilotError("Synthetic durable resume repeated a construction")
        y = np.asarray([0, 1, 0, 1], dtype=np.int8)
        metrics = classification_metrics(y, resumed["arrays"]["candidate_score"],
                                         resumed["arrays"]["candidate_class"])
        final_events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
        if sum(e.get("event") == "PREDICTOR_STARTED" for e in final_events) != 1:
            raise PilotError("Synthetic lifecycle recorded a duplicate start")
        return {
            "status": "PASS_SYNTHETIC_END_TO_END",
            "production_paths_written": False,
            "query_target_in_candidate_bundle": False,
            "first_status": first["status"],
            "resume_status": resumed["status"],
            "build_calls": build_calls["n"],
            "output_member_count": len(resumed["arrays"]),
            "candidate_metrics_after_persistence": metrics,
            "ledger_events": [e["event"] for e in final_events],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("synthetic-self-test", "prepare-freeze", "initialize-budget", "run-unit", "run-all"))
    parser.add_argument("--origin", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    try:
        if args.command == "synthetic-self-test":
            result = synthetic_lifecycle()
        elif args.command == "prepare-freeze":
            result = prepare_freeze()
        elif args.command == "initialize-budget":
            initialize_budget()
            result = {"status": "RESERVED", "units": list(FIT_IDS), "cap": 4}
        elif args.command == "run-unit":
            if args.origin is None or args.arm is None:
                raise PilotError("run-unit requires --origin and --arm")
            result = run_unit(args.origin, args.arm)
        else:
            result = run_all()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
