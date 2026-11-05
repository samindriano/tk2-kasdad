"""Staged fixed-policy evaluation for CLS_POS_FREQUENCY_RIDGE_V1.

The source-freeze and prediction stage are target-free. Evaluation targets are
read only by --evaluate after the frozen prediction manifest is committed.
This script has no Test, SampleSubmission, Kaggle, or fitting path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr
from src.models.pos_frequency_ridge_v1 import canonical_sha256, sha256_array

REL_FREEZE_R1 = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R1.json"
REL_FREEZE_R2 = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R2.json"
REL_FREEZE_R3 = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R3.json"
REL_FREEZE = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R4.json"
REL_PREDICTIONS = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_eval_v1/predictions_r1.npz"
REL_PREDICTION_MANIFEST = "reports/optimization_v3/cls_pos_frequency_ridge_v1/PREDICTIONS_MANIFEST_R1.json"
REL_RESULTS = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESULTS_R1.json"
REL_REPORT = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_REPORT_R1.md"
REL_SOURCE_QA_R1 = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_SOURCE_QA_R1.json"
REL_SOURCE_QA_R2 = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_SOURCE_QA_R2.json"
REL_SOURCE_QA_R3 = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_SOURCE_QA_R3.json"
REL_SOURCE_QA = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_SOURCE_QA_R4.json"
RESOURCE_AMENDMENT_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_RESOURCE_AMENDMENT_R4.json"
R2_FREEZE_COMMIT = "648ee016b63afddedebedd0888e889005d3b45f3"
R3_FREEZE_COMMIT = "2f712755ea3af077d17f283487f20b405c3dce31"
QUERY_INITIAL_MIN_BYTES = 3 * 1024**3
QUERY_INCREMENTAL_ESTIMATE_BYTES = 512 * 1024**2
QUERY_PROJECTED_MIN_BYTES = 2 * 1024**3

TRAIN_REL = "data/raw/classification/Train.csv"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
A02_REL = "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
A02_SHA256 = "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351"
INCIDENT_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_ACCESS_INCIDENT_01.json"
HANDOFF_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/SOL_HANDOFF.md"
PROTOCOL_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/protocol.json"
ROOT_PASS_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/ROOT_SOURCE_PASS.json"
SOURCE_PREFLIGHT_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/SOURCE_PREFLIGHT_R2.json"
POS_BAG_ROOT = "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
PILOT_ROOT = "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
POS_BAG_FREEZE_REL = f"{POS_BAG_ROOT}/FREEZE.json"
PILOT_FREEZE_REL = f"{PILOT_ROOT}/FREEZE.json"
PILOT_ACCOUNTING_REL = "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json"
A02_FREEZE_REL = "reports/optimization_v3/cls_a02_missing_temporal_decision_v2/PREFIT_FREEZE.json"

ORIGINS = {67: {"rows": 20903, "cycle_first": 68, "cycle_last": 79},
           75: {"rows": 8491, "cycle_first": 76, "cycle_last": 79}}
VIEWS = {
    "E67_H12": (67, 68, 79), "E67_W4": (67, 68, 71),
    "E67_W5": (67, 72, 75), "E67_W6": (67, 76, 79),
    "E75_W6": (75, 76, 79),
}
GROUPS = ("availability", "entry_friction", "text_load")
SUPPORT_N = 200
SUPPORT_CLASS = 30
LOGLOSS_EPS = 1e-15
OPERATIONAL_MISSING_THRESHOLD = 0.53230976129477

POS_KEYS = ("score_rows", "score_record_ids", "score_cycles", "present_mask",
            "prob_bag", "prob_offset_real", "prob_missing_hgb")
PILOT_KEYS = ("score_rows", "score_record_ids", "score_cycles", "present_mask", "prob_bag")
A02_KEYS = tuple(
    f"E{o}_{key}"
    for o in (67, 75)
    for key in ("rows", "ids", "cycles", "pA", "pP", "b", "missing", "MIX50",
                "group_availability", "group_entry_friction", "group_text_load")
)
FORBIDDEN_QUERY_KEYS = frozenset({"E67_y", "E75_y"})

SOURCE_FILES = (
    HANDOFF_REL, INCIDENT_REL, PROTOCOL_REL, ROOT_PASS_REL, SOURCE_PREFLIGHT_REL,
    POS_BAG_FREEZE_REL, PILOT_FREEZE_REL, PILOT_ACCOUNTING_REL, A02_FREEZE_REL,
    "src/models/pos_frequency_ridge_v1.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/optimization_v3_posoffset_a02_endgame_v1.py",
    "scripts/evaluate_optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_frequency_ridge_eval_v1.py",
    "tests/test_pos_frequency_ridge_v1.py",
    "tests/test_cls_pos_frequency_ridge_evaluation_v1.py",
    RESOURCE_AMENDMENT_REL,
)


class EvaluationError(RuntimeError):
    """Raised when a frozen evaluation input or boundary check fails."""


def path(rel: str) -> Path:
    return ROOT / rel


def file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object: {file_path}")
    return value


def write_json_once(file_path: Path, value: Mapping[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if file_path.exists():
        if file_path.read_text(encoding="utf-8") != encoded:
            raise EvaluationError(f"Refusing to overwrite immutable evidence: {file_path}")
        return
    with file_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def git_text(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise EvaluationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def require_committed(rel: str) -> str:
    try:
        committed_blob = git_text("rev-parse", f"HEAD:{rel}")
    except EvaluationError as exc:
        raise EvaluationError(f"Required checkpoint is not committed: {rel}") from exc
    worktree_blob = git_text("hash-object", rel)
    if committed_blob != worktree_blob:
        raise EvaluationError(f"Working file differs from committed checkpoint: {rel}")
    return git_text("log", "-1", "--format=%H", "--", rel)


def read_allowlisted_members(archive: Any, requested: Sequence[str],
                             allowed: Sequence[str]) -> dict[str, np.ndarray]:
    """Validate requested keys before any archive member is indexed."""
    names = tuple(requested)
    allowed_set = frozenset(allowed)
    forbidden = [name for name in names if name in FORBIDDEN_QUERY_KEYS or name.endswith("_y")]
    if forbidden:
        raise EvaluationError(f"Target member is forbidden in pre-label stage: {forbidden}")
    if len(names) != len(set(names)) or not set(names).issubset(allowed_set):
        raise EvaluationError("Requested NPZ fields are duplicated or outside the explicit allowlist")
    available = set(archive.files)
    missing = set(names) - available
    if missing:
        raise EvaluationError(f"Required allowlisted NPZ fields are absent: {sorted(missing)}")
    return {name: np.asarray(archive[name]) for name in names}


def safe_npz_load(file_path: Path, requested: Sequence[str],
                  allowed: Sequence[str]) -> dict[str, np.ndarray]:
    if set(allowed) & FORBIDDEN_QUERY_KEYS or any(name.endswith("_y") for name in allowed):
        raise EvaluationError("An NPZ allowlist itself contains a target key")
    with np.load(file_path, allow_pickle=False) as archive:
        return read_allowlisted_members(archive, requested, allowed)


def expected_input_paths() -> list[str]:
    rels = {TRAIN_REL, A02_REL}
    for origin in (67, 75):
        rels.update({
            f"{POS_BAG_ROOT}/E{origin}/predictions_E{origin}.npz",
            f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz",
            f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json",
            f"{PILOT_ROOT}/E{origin}/predictions_E{origin}.npz",
            f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json",
            f"{PILOT_ROOT}/units/POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR.joblib",
            f"{PILOT_ROOT}/receipts/POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR.json",
            f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/E{origin}_ALIGNED.npz",
            f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/E{origin}_PERMUTED.npz",
            f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/E{origin}_ALIGNED.json",
            f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/E{origin}_PERMUTED.json",
            f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.npz",
            f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.json",
        })
        manifest = read_json(path(f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json"))
        for key in ("score_base", "score_position"):
            rels.add(str(manifest["matrices"][key]["path"]))
    rels.update(SOURCE_FILES)
    rels.add(REL_FREEZE_R1)
    rels.add(REL_FREEZE_R2)
    rels.add(REL_FREEZE_R3)
    rels.add(REL_SOURCE_QA_R1)
    rels.add(REL_SOURCE_QA_R2)
    rels.add(REL_SOURCE_QA_R3)
    rels.add(REL_SOURCE_QA)
    rels.add("reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json")
    rels.add("experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/BUDGET_LEDGER.jsonl")
    rels.update({
        f"{POS_BAG_ROOT}/PREDICTIONS_READY.json",
        f"{POS_BAG_ROOT}/PRELABEL_VERIFICATION.json",
    })
    for rel in rels:
        if "Test" in rel or "SampleSubmission" in rel or "submission" in rel.lower():
            raise EvaluationError(f"Test/submission input is prohibited: {rel}")
    return sorted(rels)


def source_preflight() -> dict[str, Any]:
    prior_freeze = read_json(path(REL_FREEZE_R1))
    prior_body = dict(prior_freeze)
    prior_claim = prior_body.pop("freeze_sha256", None)
    if (prior_claim != canonical_sha256(prior_body)
            or prior_freeze.get("status") != "FROZEN_BEFORE_QUERY_TARGET_LOAD"):
        raise EvaluationError("Prior committed R1 source-freeze lineage is invalid")
    prior_r2 = read_json(path(REL_FREEZE_R2))
    prior_r2_body = dict(prior_r2)
    prior_r2_claim = prior_r2_body.pop("freeze_sha256", None)
    if (prior_r2_claim != canonical_sha256(prior_r2_body)
            or prior_r2.get("status") != "FROZEN_BEFORE_QUERY_TARGET_LOAD"
            or prior_r2.get("freeze_round") != "R2"):
        raise EvaluationError("Prior R2 source-freeze lineage is invalid")
    prior_r2_commit = require_committed(REL_FREEZE_R2)
    if prior_r2_commit != R2_FREEZE_COMMIT:
        raise EvaluationError("Committed R2 source-freeze checkpoint differs from the recorded parent")
    prior_r3 = read_json(path(REL_FREEZE_R3))
    prior_r3_body = dict(prior_r3)
    prior_r3_claim = prior_r3_body.pop("freeze_sha256", None)
    if (prior_r3_claim != canonical_sha256(prior_r3_body)
            or prior_r3.get("status") != "FROZEN_BEFORE_QUERY_TARGET_LOAD"
            or prior_r3.get("freeze_round") != "R3"
            or prior_r3.get("supersedes_freeze") != REL_FREEZE_R2
            or prior_r3.get("parent_freeze_commit") != R2_FREEZE_COMMIT
            or prior_r3.get("parent_freeze_sha256") != file_sha256(path(REL_FREEZE_R2))):
        raise EvaluationError("Prior R3 source-freeze lineage is invalid")
    prior_r3_commit = require_committed(REL_FREEZE_R3)
    if prior_r3_commit != R3_FREEZE_COMMIT:
        raise EvaluationError("Committed R3 source-freeze checkpoint differs from the recorded parent")
    amendment = read_json(path(RESOURCE_AMENDMENT_REL))
    if (amendment.get("decision") != "AUTHORIZE_VERSIONED_EVALUATION_ONLY_RESOURCE_POLICY"
            or amendment.get("new_model_fits") != 0
            or amendment.get("minimum_available_bytes") != QUERY_INITIAL_MIN_BYTES
            or amendment.get("incremental_memory_estimate_bytes") != QUERY_INCREMENTAL_ESTIMATE_BYTES
            or amendment.get("minimum_projected_free_bytes") != QUERY_PROJECTED_MIN_BYTES):
        raise EvaluationError("R4 resource amendment contents differ from the authorized record")
    source_hashes: dict[str, str] = {}
    for rel in SOURCE_FILES:
        file_path = path(rel)
        if not file_path.is_file():
            raise EvaluationError(f"Missing frozen source: {rel}")
        source_hashes[rel] = file_sha256(file_path)
    input_hashes: dict[str, str] = {}
    for rel in expected_input_paths():
        file_path = path(rel)
        if not file_path.is_file():
            raise EvaluationError(f"Missing frozen input: {rel}")
        input_hashes[rel] = file_sha256(file_path)
    if input_hashes.get(TRAIN_REL) != TRAIN_SHA256:
        raise EvaluationError("Official Train byte hash differs from frozen source")
    if input_hashes.get(A02_REL) != A02_SHA256:
        raise EvaluationError("A02 policy bundle hash differs from its prefit freeze")
    a02_freeze = read_json(path(A02_FREEZE_REL))
    if a02_freeze.get("selection_input_hashes", {}).get(A02_REL) != A02_SHA256:
        raise EvaluationError("A02 bundle hash is not pinned in its original prefit freeze")
    root_pass = read_json(path(ROOT_PASS_REL))
    if root_pass.get("status") != "ROOT_SOURCE_PASS":
        raise EvaluationError("Original four-fit ROOT_SOURCE_PASS is absent or invalid")
    for rel, digest in root_pass.get("source_hashes", {}).items():
        if not path(rel).is_file() or file_sha256(path(rel)) != digest:
            raise EvaluationError(f"Four-fit source hash drift: {rel}")
    if not path(INCIDENT_REL).is_file():
        raise EvaluationError("Required evaluation access incident disclosure is missing")
    if any(x.endswith("_y") for x in A02_KEYS) or FORBIDDEN_QUERY_KEYS & set(A02_KEYS):
        raise EvaluationError("A02 explicit allowlist includes a target field")
    return {
        "status": "PASS_TARGET_FREE_EVALUATOR_SOURCE_PREFLIGHT",
        "source_hashes": source_hashes, "input_hashes": input_hashes,
        "parent_freeze": {
            "path": REL_FREEZE_R3, "freeze_sha256": file_sha256(path(REL_FREEZE_R3)),
            "commit": prior_r3_commit,
        },
        "a02_allowlist": list(A02_KEYS), "pos_bundle_allowlist": list(POS_KEYS),
        "pilot_bundle_allowlist": list(PILOT_KEYS),
        "query_label_arrays_materialized": False, "query_labels_scored": False,
        "test_or_samplesubmission_accessed": False, "estimator_fits": 0,
    }


def build_source_freeze(*, synthetic_qa: Mapping[str, Any]) -> dict[str, Any]:
    if synthetic_qa.get("status") != "PASS" or int(synthetic_qa.get("tests_passed", 0)) < 5:
        raise EvaluationError("Synthetic evaluator QA must pass before source freeze")
    preflight = source_preflight()
    body: dict[str, Any] = {
        "schema_version": 1, "status": "FROZEN_BEFORE_QUERY_TARGET_LOAD",
        "campaign_id": "CLS_POS_FREQUENCY_RIDGE_V1_EVALUATION_R1",
        "freeze_round": "R4",
        "supersedes_freeze": REL_FREEZE_R3,
        "supersession_reason": (
            "R4 is an explicitly authorized evaluation-only resource amendment following the R3 source repair. "
            "It changes only the zero-fit query-replay resource policy to an initial 3 GiB floor plus a "
            "projected 2 GiB reserve, adds measured current-state boundary samples, and releases origin matrices "
            "serially. Scientific recipes, inputs, labels, windows, thresholds, comparisons, and gates are unchanged."
        ),
        "source_preparation_parent_commit": git_text("rev-parse", "HEAD"),
        "parent_freeze_sha256": preflight["parent_freeze"]["freeze_sha256"],
        "parent_freeze_commit": preflight["parent_freeze"]["commit"],
        "resource_policy": {
            "amendment_path": RESOURCE_AMENDMENT_REL,
            "minimum_available_bytes": QUERY_INITIAL_MIN_BYTES,
            "incremental_memory_estimate_bytes": QUERY_INCREMENTAL_ESTIMATE_BYTES,
            "minimum_projected_free_bytes": QUERY_PROJECTED_MIN_BYTES,
            "post_load_and_origin_boundary_minimum_available_bytes": QUERY_PROJECTED_MIN_BYTES,
            "training_and_reg_lp_gates_unchanged": "minimum 4 GiB plus 2 GiB reserve",
            "sampled_rss_is_peak": False,
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": preflight["source_hashes"], "input_hashes": preflight["input_hashes"],
        "raw_file_hashing_only": {
            "paths": [TRAIN_REL, A02_REL],
            "meaning": "verify byte identity only; no target column parsed and no NPZ target member decompressed during freeze",
        },
        "field_allowlists": {
            "A02_policy_bundle": list(A02_KEYS),
            "POS_BAG_OFFSET_predictions": list(POS_KEYS),
            "POSITIONAL_TEXT_PILOT_predictions": list(PILOT_KEYS),
            "gamma_artifacts": ["gamma"],
            "query_target_keys_forbidden_prelabel": sorted(FORBIDDEN_QUERY_KEYS),
        },
        "windows": {key: list(value) for key, value in VIEWS.items()},
        "unique_pool": ["E67_W4", "E67_W5", "E75_W6"],
        "recipe": {
            "component": "ALIGNED, PERMUTED, original uniform POS gamma; shared missing HGB fallback",
            "competition_present": "0.5 * new POS probability + 0.5 * same-origin pA",
            "competition_missing_primary": "pA, threshold >= 0.50",
            "competition_missing_operational_descriptive": {
                "pA": OPERATIONAL_MISSING_THRESHOLD,
                "classification": "post-selected descriptive reference only",
            },
            "threshold_present_and_primary_missing": 0.5, "logloss_epsilon": LOGLOSS_EPS,
            "subgroup_support": {"n_min": SUPPORT_N, "class_0_min": SUPPORT_CLASS,
                                 "class_1_min": SUPPORT_CLASS},
        },
        "gates": {
            "mechanism": "ALIGNED Macro F1 > PERMUTED and > UNIFORM by >1e-12 on E67_H12 and E75_W6",
            "competition": {
                "E67_H12_gain_min": 0.0003, "E75_W6_gain_min": 0.0,
                "unique_pool_gain_min": 0.0003, "E67_window_harm_max": 0.003,
                "supported_subgroup_macro_f1_harm_max": 0.01,
                "supported_subgroup_positive_recall_harm_max": 0.02,
                "apply_on_all_reported_views": True,
            },
        },
        "incident_disclosure": INCIDENT_REL,
        "incident_status": "PRE-EXPOSURE CANNOT BE CLAIMED; repeatedly reused development targets",
        "synthetic_qa": dict(synthetic_qa),
        "target_or_test_access_during_freeze": False, "new_fits_or_test_inference": 0,
    }
    body["freeze_sha256"] = canonical_sha256(body)
    return body


def run_synthetic_evaluator_qa() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q",
               "tests/test_cls_pos_frequency_ridge_evaluation_v1.py"]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise EvaluationError("Synthetic evaluator QA failed:\n" + proc.stdout + proc.stderr)
    match = re.search(r"(\d+) passed", proc.stdout + proc.stderr)
    if not match:
        raise EvaluationError("Could not verify pytest passed-test count")
    qa = {
        "status": "PASS", "tests_passed": int(match.group(1)),
        "command": "python -m pytest -q tests/test_cls_pos_frequency_ridge_evaluation_v1.py",
        "stdout": (proc.stdout + proc.stderr).strip(),
        "query_targets_loaded": False, "official_data_loaded": False,
        "test_or_samplesubmission_accessed": False,
    }
    write_json_once(path(REL_SOURCE_QA), qa)
    return qa


def verify_source_freeze(*, require_frozen_commit: bool) -> tuple[dict[str, Any], str | None]:
    freeze = read_json(path(REL_FREEZE))
    body = dict(freeze)
    claimed = body.pop("freeze_sha256", None)
    if claimed != canonical_sha256(body):
        raise EvaluationError("Evaluation source freeze signature mismatch")
    if freeze.get("status") != "FROZEN_BEFORE_QUERY_TARGET_LOAD":
        raise EvaluationError("Evaluation source freeze status is not pre-label")
    if (freeze.get("freeze_round") != "R4"
            or freeze.get("supersedes_freeze") != REL_FREEZE_R3):
        raise EvaluationError("Evaluation source freeze is not the committed R4 resource checkpoint")
    current = source_preflight()
    if (freeze.get("parent_freeze_sha256") != current["parent_freeze"]["freeze_sha256"]
            or freeze.get("parent_freeze_commit") != current["parent_freeze"]["commit"]):
        raise EvaluationError("R3 freeze does not bind the verified R2 parent checkpoint")
    if current["source_hashes"] != freeze.get("source_hashes"):
        raise EvaluationError("Evaluator source or frozen predecessor source changed")
    if current["input_hashes"] != freeze.get("input_hashes"):
        raise EvaluationError("Evaluation input file hashes changed after source freeze")
    freeze_commit = require_committed(REL_FREEZE) if require_frozen_commit else None
    return freeze, freeze_commit


def gamma_from_artifact(artifact_rel: str, receipt_rel: str) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = path(artifact_rel); receipt_path = path(receipt_rel)
    receipt = read_json(receipt_path)
    artifact_sha = file_sha256(artifact)
    if receipt.get("artifact_sha256") != artifact_sha:
        raise EvaluationError(f"Gamma artifact byte hash differs from receipt: {artifact_rel}")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma"}:
            raise EvaluationError(f"Gamma archive schema mismatch: {artifact_rel}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64)
    if not np.isfinite(gamma).all() or sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise EvaluationError(f"Gamma content differs from receipt: {artifact_rel}")
    return gamma, {"artifact_sha256": artifact_sha, "receipt_sha256": file_sha256(receipt_path),
                   "gamma_sha256": sha256_array(gamma), "fit_id": receipt.get("fit_id"),
                   "origin": receipt.get("origin"), "arm": receipt.get("arm")}


def load_matrix(manifest: Mapping[str, Any], key: str) -> sparse.csr_matrix:
    item = manifest["matrices"][key]; file_path = path(str(item["path"]))
    if file_sha256(file_path) != item["sha256"]:
        raise EvaluationError(f"Frozen matrix byte hash mismatch for {key}")
    matrix = sparse.load_npz(file_path).tocsr()
    if (list(matrix.shape) != item["shape"] or str(matrix.dtype) != item["dtype"]
            or sha256_csr(matrix) != item["logical_sha256"]):
        raise EvaluationError(f"Frozen matrix shape/logical hash mismatch for {key}")
    return matrix


def write_npz_once(file_path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        with np.load(file_path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise EvaluationError("Existing saved prediction bundle has another schema")
            for name, value in arrays.items():
                if not np.array_equal(archive[name], value):
                    raise EvaluationError(f"Existing saved predictions differ: {name}")
        return
    with file_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()


def evaluate_query_initial_resource_gates(available_bytes: int,
                                          estimated_additional_bytes: int = QUERY_INCREMENTAL_ESTIMATE_BYTES
                                          ) -> dict[str, Any]:
    available = int(available_bytes)
    estimate = int(estimated_additional_bytes)
    projected = available - estimate
    initial_pass = available >= QUERY_INITIAL_MIN_BYTES
    projected_pass = projected >= QUERY_PROJECTED_MIN_BYTES
    return {
        "available_bytes": available,
        "estimated_additional_bytes": estimate,
        "projected_free_bytes": projected,
        "minimum_available_bytes": QUERY_INITIAL_MIN_BYTES,
        "minimum_projected_free_bytes": QUERY_PROJECTED_MIN_BYTES,
        "initial_floor_pass": initial_pass,
        "projected_reserve_pass": projected_pass,
        "pass_both_initial_gates": initial_pass and projected_pass,
        "estimate_is_measured_peak": False,
    }


def evaluate_query_boundary_resource_gate(available_bytes: int) -> dict[str, Any]:
    available = int(available_bytes)
    return {
        "available_bytes": available,
        "minimum_available_bytes": QUERY_PROJECTED_MIN_BYTES,
        "pass": available >= QUERY_PROJECTED_MIN_BYTES,
        "sample_is_peak": False,
    }


def resource_snapshot() -> dict[str, Any]:
    import psutil
    available = int(psutil.virtual_memory().available)
    process_rss = int(psutil.Process().memory_info().rss)
    return {
        "policy_id": "CLS_EVALUATION_QUERY_REPLAY_R4",
        **evaluate_query_initial_resource_gates(available),
        "process_rss_sample_bytes": process_rss,
        "sample_kind": "CURRENT_STATE_SNAPSHOT_NOT_PEAK",
    }


def resource_boundary_snapshot(stage: str) -> dict[str, Any]:
    import psutil
    available = int(psutil.virtual_memory().available)
    process_rss = int(psutil.Process().memory_info().rss)
    return {
        "stage": stage,
        **evaluate_query_boundary_resource_gate(available),
        "process_rss_sample_bytes": process_rss,
        "sample_kind": "CURRENT_STATE_SNAPSHOT_NOT_PEAK",
    }


class SharedJobLock:
    """Nonblocking advisory lock shared with the active Regression lane."""

    def __init__(self) -> None:
        import os
        import tempfile
        self.file = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
        self.stream: Any = None
        self.unlock: Any = None
        self.os = os

    def __enter__(self) -> "SharedJobLock":
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.file.open("a+b")
        self.stream.seek(0, self.os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0"); self.stream.flush()
        self.stream.seek(0)
        if self.os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                self.stream.close()
                raise EvaluationError("Another KASDAD heavy job holds shared slot; stop without waiting/polling") from exc
            self.unlock = lambda: msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self.stream.close()
                raise EvaluationError("Another KASDAD heavy job holds shared slot; stop without waiting/polling") from exc
            self.unlock = lambda: fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.stream is not None:
            try:
                self.stream.seek(0)
                self.unlock()
            finally:
                self.stream.close()


def row_metadata(rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray) -> dict[str, Any]:
    if len(rows) == 0 or rows.shape != ids.shape or rows.shape != cycles.shape:
        raise EvaluationError("Row metadata vectors are empty or misaligned")
    return {
        "rows": int(len(rows)), "cycle_min": int(cycles.min()), "cycle_max": int(cycles.max()),
        "row_indices_sha256": sha256_array(rows.astype(np.int64)),
        # Matrix manifests hash pandas StringDtype -> astype(str) -> object arrays.
        # Preserve that historical dtype in the digest while callers separately
        # enforce elementwise exact ordered Unicode ID equality.
        "record_ids_sha256": sha256_array(ids.astype(object)),
        "cycles_sha256": sha256_array(cycles.astype(np.int64)),
    }


def validate_ordered_row_identity(rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
                                  expected_rows: np.ndarray, expected_ids: np.ndarray,
                                  expected_cycles: np.ndarray) -> None:
    """Require exact ordered row/ID/cycle equality and unique physical identities."""
    rows = np.asarray(rows, dtype=np.int64)
    ids = np.asarray(ids).astype(str)
    cycles = np.asarray(cycles, dtype=np.int64)
    expected_rows = np.asarray(expected_rows, dtype=np.int64)
    expected_ids = np.asarray(expected_ids).astype(str)
    expected_cycles = np.asarray(expected_cycles, dtype=np.int64)
    if (rows.ndim != 1 or ids.ndim != 1 or cycles.ndim != 1 or len(rows) == 0
            or rows.shape != ids.shape or rows.shape != cycles.shape
            or expected_rows.shape != rows.shape or expected_ids.shape != rows.shape
            or expected_cycles.shape != rows.shape):
        raise EvaluationError("Ordered row identity vectors are empty or misaligned")
    if (len(np.unique(rows)) != len(rows) or len(np.unique(ids)) != len(ids)
            or not np.array_equal(rows, expected_rows)
            or not np.array_equal(ids, expected_ids)
            or not np.array_equal(cycles, expected_cycles)):
        raise EvaluationError("Ordered physical rows, record IDs, or cycles differ from the frozen population")


def verify_four_fit_ledger() -> dict[str, Any]:
    ledger = path("experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/BUDGET_LEDGER.jsonl")
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    fit_ids = {"E67_ALIGNED", "E67_PERMUTED", "E75_ALIGNED", "E75_PERMUTED"}
    starts = [x for x in events if x.get("event") == "FIT_STARTED"]
    saved = [x for x in events if x.get("event") == "FIT_SAVED"]
    failed = [x for x in events if x.get("event") == "FIT_FAILED"]
    started_ids = [x.get("fit_id") for x in starts]
    saved_ids = [x.get("fit_id") for x in saved]
    if (len(starts) != 4 or set(started_ids) != fit_ids or len(set(started_ids)) != 4
            or len(saved) != 4 or set(saved_ids) != fit_ids or len(set(saved_ids)) != 4 or failed):
        raise EvaluationError("Four-fit ledger does not show exactly four unique saved fits")
    for fit_id in fit_ids:
        i_start = next(i for i,x in enumerate(events) if x.get("event") == "FIT_STARTED" and x.get("fit_id") == fit_id)
        i_saved = next(i for i,x in enumerate(events) if x.get("event") == "FIT_SAVED" and x.get("fit_id") == fit_id)
        if i_start >= i_saved:
            raise EvaluationError(f"Fit ledger saved receipt precedes its start: {fit_id}")
        receipt_path = path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/{fit_id}.json")
        artifact_path = path(f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/{fit_id}.npz")
        event = events[i_saved]
        if (event.get("receipt_sha256") != file_sha256(receipt_path)
                or event.get("artifact_sha256") != file_sha256(artifact_path)):
            raise EvaluationError(f"Saved ledger event does not bind fit receipt/artifact: {fit_id}")
    return {"status": "PASS_EXACTLY_FOUR_SAVED_NO_FAILURES",
            "starts": len(starts), "saved": len(saved), "failed": len(failed),
            "fit_ids": sorted(fit_ids), "ledger_sha256": file_sha256(ledger)}


def prepare_predictions() -> dict[str, Any]:
    with SharedJobLock():
        initial_resource = resource_snapshot()
        if not initial_resource["pass_both_initial_gates"]:
            raise EvaluationError(
                "R4 initial RAM gates failed before target-free prediction replay: "
                f"{initial_resource}"
            )
        freeze, freeze_commit = verify_source_freeze(require_frozen_commit=True)
        resource_samples: list[dict[str, Any]] = [
            {"stage": "INITIAL_UNDER_SHARED_LOCK", **initial_resource}
        ]
        protocol = read_json(path(PROTOCOL_REL))
        root_pass = read_json(path(ROOT_PASS_REL))
        if protocol.get("campaign_id") != "CLS_POS_FREQUENCY_RIDGE_V1":
            raise EvaluationError("Unexpected frozen fit protocol")
        ledger_audit = verify_four_fit_ledger()
        a02_freeze = read_json(path(A02_FREEZE_REL))
        if a02_freeze.get("selection_input_hashes", {}).get(A02_REL) != A02_SHA256:
            raise EvaluationError("A02 source is not the pinned fixed probability bundle")

        # Archive directory/header inspection does not read compressed member
        # data. Explicit keys below exclude E67_y and E75_y.
        with zipfile.ZipFile(path(A02_REL)) as archive:
            names = {info.filename[:-4] for info in archive.infolist() if info.filename.endswith(".npy")}
        if set(A02_KEYS) - names or set(A02_KEYS) & FORBIDDEN_QUERY_KEYS:
            raise EvaluationError("A02 target-free allowlist/header check failed")
        if any(name.endswith("_y") for name in A02_KEYS):
            raise EvaluationError("A02 query-target member requested before saved predictions")

        train_path = path(TRAIN_REL)
        if file_sha256(train_path) != TRAIN_SHA256:
            raise EvaluationError("Official Train hash differs before identity-only read")
        identity = pd.read_csv(train_path, usecols=["record_id", "cycle_code"],
                               dtype={"record_id": "string"})
        if (len(identity) != 50608 or identity.record_id.isna().any()
                or identity.record_id.duplicated().any()):
            raise EvaluationError("Official Train identity schema/count failed before predictions")
        cycles_train = pd.to_numeric(identity.cycle_code, errors="raise").to_numpy(dtype=np.int64)
        ids_train = identity.record_id.astype(str).to_numpy()
        if cycles_train.min() != 47 or cycles_train.max() != 79:
            raise EvaluationError("Official Train cycle coverage differs from frozen 47-79")
        a02 = safe_npz_load(path(A02_REL), A02_KEYS, A02_KEYS)
        if FORBIDDEN_QUERY_KEYS & set(a02):
            raise EvaluationError("A02 target keys were materialized in the pre-label stage")

        outputs: dict[str, np.ndarray] = {}
        lineages: dict[str, Any] = {}
        replay_audit: dict[str, Any] = {}
        pos_freeze = read_json(path(POS_BAG_FREEZE_REL))
        pos_ready = read_json(path(f"{POS_BAG_ROOT}/PREDICTIONS_READY.json"))
        pos_prelabeled = read_json(path(f"{POS_BAG_ROOT}/PRELABEL_VERIFICATION.json"))
        if (pos_ready.get("freeze_signature") != pos_freeze.get("freeze_signature")
                or pos_ready.get("all_four_gamma_receipts_durable") is not True
                or pos_ready.get("query_labels_opened") is not False
                or pos_prelabeled.get("status") != "PASS"
                or pos_prelabeled.get("query_labels_read") is not False):
            raise EvaluationError("Original POS bundle readiness/prefix-only receipt failed")
        accounting = read_json(path(PILOT_ACCOUNTING_REL))
        for origin in (67, 75):
            key = f"E{origin}"
            expected = ORIGINS[origin]
            manifest_rel = f"{PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json"
            manifest = read_json(path(manifest_rel))
            manifest_sha = file_sha256(path(manifest_rel))
            if manifest_sha != pos_freeze["origins"][key]["matrix_manifest_sha256"]:
                raise EvaluationError(f"E{origin} matrix manifest differs from POS freeze")
            pos_rel = f"{POS_BAG_ROOT}/{key}/predictions_{key}.npz"
            pilot_rel = f"{PILOT_ROOT}/E{origin}/predictions_E{origin}.npz"
            pos = safe_npz_load(path(pos_rel), POS_KEYS, POS_KEYS)
            pilot = safe_npz_load(path(pilot_rel), PILOT_KEYS, PILOT_KEYS)

            rows = a02[f"E{origin}_rows"].astype(np.int64)
            ids = a02[f"E{origin}_ids"].astype(str)
            cycles = a02[f"E{origin}_cycles"].astype(np.int64)
            expected_rows = np.flatnonzero((cycles_train >= expected["cycle_first"])
                                           & (cycles_train <= expected["cycle_last"]))
            expected_ids = ids_train[expected_rows]
            expected_cycles = cycles_train[expected_rows]
            if len(expected_rows) != expected["rows"]:
                raise EvaluationError(f"E{origin} A02 query rows/IDs/cycles differ from official Train")
            validate_ordered_row_identity(rows, ids, cycles, expected_rows, expected_ids, expected_cycles)
            for source_name, source_rows, source_ids, source_cycles in (
                ("POS_BAG", pos["score_rows"], pos["score_record_ids"].astype(str), pos["score_cycles"]),
                ("POSITIONAL_PILOT", pilot["score_rows"], pilot["score_record_ids"].astype(str), pilot["score_cycles"]),
            ):
                try:
                    validate_ordered_row_identity(source_rows, source_ids, source_cycles, rows, ids, cycles)
                except EvaluationError as exc:
                    raise EvaluationError(f"E{origin} {source_name} ordered query identity differs")
            if row_metadata(rows, ids, cycles) != manifest.get("score_row_metadata"):
                raise EvaluationError(f"E{origin} current query IDs/cycles differ from matrix manifest")

            missing = a02[f"E{origin}_missing"].astype(bool)
            present = ~missing
            if (not np.array_equal(pos["present_mask"].astype(bool), present)
                    or not np.array_equal(pilot["present_mask"].astype(bool), present)):
                raise EvaluationError(f"E{origin} stream-availability route does not align")
            p_a = a02[f"E{origin}_pA"].astype(np.float64)
            missing_hgb = pos["prob_missing_hgb"].astype(np.float64)
            p_bag_saved = pilot["prob_bag"].astype(np.float64)
            if not np.array_equal(missing_hgb[missing], p_bag_saved[missing]):
                raise EvaluationError(f"E{origin} HGB fallback does not match saved pilot cache")
            p_offset_cached = pos["prob_offset_real"].astype(np.float64)
            if any(not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
                   for x in (p_a, missing_hgb, p_offset_cached)):
                raise EvaluationError(f"E{origin} saved probabilities are invalid")

            score_base = load_matrix(manifest, "score_base")
            score_position_augmented = load_matrix(manifest, "score_position")
            if (score_base.shape[0] != len(rows)
                    or score_position_augmented.shape[0] != len(rows)
                    or score_position_augmented.shape[1] <= score_base.shape[1]):
                raise EvaluationError(f"E{origin} query matrices do not match frozen row count")
            if (score_position_augmented[:, :score_base.shape[1]] != score_base).nnz:
                raise EvaluationError(f"E{origin} augmented query matrix changed the frozen C2_B base block")
            load_sample = resource_boundary_snapshot(f"E{origin}_AFTER_MATRIX_LOAD_AND_IDENTITY_BOUNDARIES")
            resource_samples.append(load_sample)
            if not load_sample["pass"]:
                raise EvaluationError(
                    f"R4 actual available RAM fell below 2 GiB after E{origin} matrix load: {load_sample}"
                )
            score_position = score_position_augmented[:, score_base.shape[1]:].tocsr()
            del score_position_augmented
            bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
            bag_entry = next((unit for unit in accounting["fit_units"]
                              if unit.get("fit_id") == bag_id), None)
            if bag_entry is None:
                raise EvaluationError(f"Missing frozen C2_B bag accounting for E{origin}")
            bag_rel = f"{PILOT_ROOT}/units/{bag_id}.joblib"
            bag_receipt_rel = f"{PILOT_ROOT}/receipts/{bag_id}.json"
            bag_path = path(bag_rel); bag_receipt_path = path(bag_receipt_rel)
            if (file_sha256(bag_path) != bag_entry["artifact_sha256"]
                    or file_sha256(bag_receipt_path) != bag_entry["receipt_sha256"]):
                raise EvaluationError(f"E{origin} saved bag estimator/receipt hash changed")
            bag_receipt = read_json(bag_receipt_path)
            if bag_receipt.get("fit_id") != bag_id or bag_receipt.get("artifact_sha256") != bag_entry["artifact_sha256"]:
                raise EvaluationError(f"E{origin} C2_B receipt does not bind the saved estimator")
            bag_unit = joblib.load(bag_path)
            bag = bag_unit.get("estimator")
            if bag is None or list(np.asarray(bag.classes_).tolist()) != [0, 1]:
                raise EvaluationError(f"E{origin} saved C2_B estimator classes are invalid")
            coef = np.asarray(bag.coef_, dtype=np.float64)
            intercept = np.asarray(bag.intercept_, dtype=np.float64).reshape(-1)
            if coef.shape != (1, score_base.shape[1]) or intercept.shape != (1,):
                raise EvaluationError(f"E{origin} C2_B estimator width differs from saved query matrix")
            z0 = np.asarray(score_base @ coef[0], dtype=np.float64).reshape(-1) + float(intercept[0])
            p_bag_replay = expit(z0)
            if not np.allclose(p_bag_replay, p_bag_saved, rtol=0.0, atol=2e-12):
                raise EvaluationError(f"E{origin} original C2_B query logits fail independent replay")

            uniform_rel = f"{POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz"
            uniform_receipt_rel = f"{POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json"
            uniform_gamma, uniform_lineage = gamma_from_artifact(uniform_rel, uniform_receipt_rel)
            uniform_receipt = read_json(path(uniform_receipt_rel))
            saved_real = pos_prelabeled["origins"][key]["fits"]["REAL"]
            ready_bundle = pos_ready["predictions"][key]
            if (uniform_receipt.get("fit_id") != f"POS_BAG_OFFSET_V1_E{origin}_REAL"
                    or uniform_receipt.get("origin") != origin or uniform_receipt.get("arm") != "REAL"
                    or uniform_receipt.get("freeze_signature") != pos_freeze.get("freeze_signature")
                    or uniform_receipt.get("gamma_sha256") != saved_real.get("gamma_sha256")
                    or file_sha256(path(uniform_receipt_rel)) != saved_real.get("receipt_sha256")
                    or file_sha256(path(pos_rel)) != ready_bundle.get("sha256")):
                raise EvaluationError(f"E{origin} original uniform POS receipt/cache lineage failed")

            gamma_by_arm: dict[str, np.ndarray] = {}
            new_lineages: dict[str, Any] = {}
            fit_rows = np.flatnonzero(cycles_train <= origin)
            fit_ids_for_receipt = identity.iloc[fit_rows]["record_id"].astype(str).to_numpy()
            fit_ids_receipt_sha = sha256_array(np.asarray(fit_ids_for_receipt).astype(str))
            for arm in ("ALIGNED", "PERMUTED"):
                artifact_rel = f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/E{origin}_{arm}.npz"
                receipt_rel = f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/E{origin}_{arm}.json"
                map_rel = f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.json"
                gamma, lineage = gamma_from_artifact(artifact_rel, receipt_rel)
                receipt = read_json(path(receipt_rel))
                if (receipt.get("campaign_id") != "CLS_POS_FREQUENCY_RIDGE_V1"
                        or receipt.get("fit_id") != f"E{origin}_{arm}"
                        or receipt.get("origin") != origin or receipt.get("arm") != arm
                        or receipt.get("source_commit") != root_pass.get("source_commit")
                        or receipt.get("root_source_pass_sha256") != file_sha256(path(ROOT_PASS_REL))
                        or receipt.get("source_hashes") != root_pass.get("source_hashes")
                        or receipt.get("fit_rows") != (29705 if origin == 67 else 42117)
                        or receipt.get("fit_ids_sha256") != fit_ids_receipt_sha
                        or receipt.get("fit_row_indices_sha256") != manifest["fit_row_metadata"]["row_indices_sha256"]
                        or receipt.get("fit_base_sha256") != manifest["matrices"]["fit_base"]["logical_sha256"]
                        or receipt.get("fit_position_sha256") != manifest["position_audit"]["fit_original"]["block_sha256"]
                        or receipt.get("fixed_bag_artifact_sha256") != bag_entry["artifact_sha256"]
                        or receipt.get("fixed_bag_receipt_sha256") != bag_entry["receipt_sha256"]
                        or receipt.get("fit_target_sha256") != uniform_receipt["fit_population"]["fit_target_sha256"]
                        or receipt.get("penalty_map_receipt_sha256") != file_sha256(path(map_rel))):
                    raise EvaluationError(f"E{origin}_{arm} fit receipt lineage differs from approved fit")
                gamma_by_arm[arm] = gamma; new_lineages[arm] = lineage
            all_gammas = [uniform_gamma, *gamma_by_arm.values()]
            if any(len(gamma) != score_position.shape[1] or not np.isfinite(gamma).all()
                   for gamma in all_gammas):
                raise EvaluationError(f"E{origin} gamma width/finiteness check failed")

            replay: dict[str, np.ndarray] = {}
            for arm, gamma in [("UNIFORM", uniform_gamma), *gamma_by_arm.items()]:
                logits = z0 + np.asarray(score_position @ gamma, dtype=np.float64).reshape(-1)
                probability = missing_hgb.copy()
                probability[present] = expit(logits[present])
                if not np.array_equal(probability[missing], missing_hgb[missing]):
                    raise EvaluationError(f"E{origin} {arm} changed missing fallback")
                replay[arm] = probability
            p_uniform = missing_hgb.copy()
            p_uniform[present] = p_offset_cached[present]
            uniform_error = float(np.max(np.abs(replay["UNIFORM"][present]-p_uniform[present])))
            if uniform_error > 2e-10:
                raise EvaluationError(f"E{origin} uniform gamma does not reproduce saved POS probabilities")
            if not np.array_equal(p_uniform[missing], missing_hgb[missing]):
                raise EvaluationError(f"E{origin} cached uniform POS is not using exact HGB fallback")
            p_pos_a02 = a02[f"E{origin}_pP"].astype(np.float64)
            if not np.array_equal(p_pos_a02, p_uniform):
                raise EvaluationError(f"E{origin} A02 pP differs from original POS+missing route")
            mix50_saved = a02[f"E{origin}_MIX50"].astype(np.float64)
            mix50_exact = 0.5*p_pos_a02 + 0.5*p_a
            if not np.allclose(mix50_saved, mix50_exact, rtol=0.0, atol=2e-15):
                raise EvaluationError(f"E{origin} cached original MIX50 fails formula replay")
            mix50_primary = mix50_exact.copy()
            mix50_primary[missing] = p_a[missing]

            groups = {
                "availability": a02[f"E{origin}_group_availability"].astype(str),
                "entry_friction": a02[f"E{origin}_group_entry_friction"].astype(str),
                "text_load": a02[f"E{origin}_group_text_load"].astype(str),
            }
            if not np.array_equal(groups["availability"], np.where(missing, "missing", "present")):
                raise EvaluationError(f"E{origin} availability group array differs from route mask")
            if any(g.shape != (len(rows),) for g in groups.values()):
                raise EvaluationError(f"E{origin} subgroup feature arrays are misaligned")
            if not set(np.unique(groups["text_load"])).issubset({"Q1", "Q2", "Q3", "Q4"}):
                raise EvaluationError(f"E{origin} fit-defined text-load groups are malformed")

            outputs.update({
                f"E{origin}_rows": rows.astype(np.int64), f"E{origin}_ids": ids.astype(str),
                f"E{origin}_cycles": cycles.astype(np.int16), f"E{origin}_missing": missing.astype(bool),
                f"E{origin}_present": present.astype(bool), f"E{origin}_pA": p_a,
                f"E{origin}_pP_UNIFORM": p_uniform, f"E{origin}_pP_ALIGNED": replay["ALIGNED"],
                f"E{origin}_pP_PERMUTED": replay["PERMUTED"], f"E{origin}_pMISSING_HGB": missing_hgb,
                f"E{origin}_MIX50_REFERENCE": mix50_primary,
                **{f"E{origin}_group_{family}": value for family, value in groups.items()},
            })
            lineages[key] = {
                "rows": int(len(rows)), "present_rows": int(present.sum()), "missing_rows": int(missing.sum()),
                "cycle_range": [int(cycles.min()), int(cycles.max())],
                "matrix_manifest_sha256": manifest_sha,
                "base_matrix_sha256": manifest["matrices"]["score_base"]["sha256"],
                "position_matrix_sha256": manifest["matrices"]["score_position"]["sha256"],
                "pos_bundle_sha256": file_sha256(path(pos_rel)),
                "pilot_bundle_sha256": file_sha256(path(pilot_rel)),
                "bag_artifact_sha256": file_sha256(bag_path),
                "bag_receipt_sha256": file_sha256(bag_receipt_path),
                "uniform_gamma": uniform_lineage, "frequency_gammas": new_lineages,
                "row_id_cycle_identity": "PASS_EXACT_ORDERED_ID_AND_CYCLE_ALIGNMENT",
                "original_c2b_query_logit_replay": "PASS_ATOL_2E-12",
                "uniform_pos_gamma_replay": "PASS_ATOL_2E-10_PRESENT_ONLY",
                "missing_fallback": "BIT_EXACT_SHARED_HGB",
            }
            replay_audit[key] = {
                "c2b_max_abs_probability_error": float(np.max(np.abs(p_bag_replay-p_bag_saved))),
                "uniform_pos_max_abs_present_error": uniform_error,
                "saved_mix50_formula_max_abs_error": float(np.max(np.abs(mix50_saved-mix50_exact))),
            }
            boundary_sample = resource_boundary_snapshot(f"E{origin}_ORIGIN_BOUNDARY_BEFORE_RELEASE")
            resource_samples.append(boundary_sample)
            if not boundary_sample["pass"]:
                raise EvaluationError(
                    f"R4 actual available RAM fell below 2 GiB at E{origin} origin boundary: {boundary_sample}"
                )
            # Retain only compact predictions/lineage in outputs. Free matrix,
            # estimator, and replay intermediates before loading the next origin.
            del (score_base, score_position, bag_unit, bag, coef, intercept, z0, p_bag_replay,
                 uniform_gamma, uniform_receipt, gamma_by_arm, new_lineages, all_gammas,
                 replay, p_uniform, p_pos_a02, mix50_saved, mix50_exact, mix50_primary,
                 groups, pos, pilot, source_rows, source_ids, source_cycles, rows, ids, cycles,
                 expected_rows, expected_ids, expected_cycles, missing, present, p_a,
                 missing_hgb, p_bag_saved, p_offset_cached, fit_rows, fit_ids_for_receipt,
                 fit_ids_receipt_sha, uniform_lineage, saved_real, ready_bundle, manifest,
                 bag_entry, bag_rel, bag_receipt_rel, bag_path, bag_receipt_path, key, expected,
                 manifest_rel, manifest_sha, pos_rel, pilot_rel, uniform_rel, uniform_receipt_rel,
                 artifact_rel, receipt_rel, map_rel, gamma, lineage, receipt, arm, logits,
                probability, source_name, boundary_sample, load_sample)
            import gc
            gc.collect()

        final_sample = resource_boundary_snapshot("AFTER_ALL_ORIGINS_RELEASED_BEFORE_SAVE")
        resource_samples.append(final_sample)
        if not final_sample["pass"]:
            raise EvaluationError(
                f"R4 actual available RAM fell below 2 GiB before prediction save: {final_sample}"
            )
        pred_path = path(REL_PREDICTIONS)
        write_npz_once(pred_path, outputs)
        pred_manifest: dict[str, Any] = {
            "schema_version": 1, "status": "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ",
            "source_freeze_commit": freeze_commit,
            "source_freeze_sha256": file_sha256(path(REL_FREEZE)),
            "source_hashes": freeze["source_hashes"], "input_hashes": freeze["input_hashes"],
            "prediction_path": REL_PREDICTIONS, "prediction_sha256": file_sha256(pred_path),
            "prediction_array_hashes": {k: sha256_array(v) for k,v in outputs.items()},
            "prediction_schema": list(outputs), "lineages": lineages,
            "replay_audit": replay_audit,
            "a02_allowlist": list(A02_KEYS), "forbidden_target_fields_requested": [],
            "four_fit_ledger_audit": ledger_audit,
            "resource_snapshot": {
                "policy_id": "CLS_EVALUATION_QUERY_REPLAY_R4",
                "initial_floor_bytes": QUERY_INITIAL_MIN_BYTES,
                "incremental_memory_estimate_bytes": QUERY_INCREMENTAL_ESTIMATE_BYTES,
                "projected_free_floor_bytes": QUERY_PROJECTED_MIN_BYTES,
                "post_load_and_boundary_floor_bytes": QUERY_PROJECTED_MIN_BYTES,
                "samples": resource_samples,
                "samples_are_peak_measurements": False,
            },
            "query_target_members_loaded": [], "query_target_labels_read": False,
            "test_or_samplesubmission_accessed": False, "estimator_fits": 0,
        }
        write_json_once(path(REL_PREDICTION_MANIFEST), pred_manifest)
    return pred_manifest


def competition_probabilities(p_pos: np.ndarray, p_a: np.ndarray,
                              present: np.ndarray) -> np.ndarray:
    p = 0.5*np.asarray(p_pos, dtype=np.float64)+0.5*np.asarray(p_a, dtype=np.float64)
    p = p.copy()
    p[~np.asarray(present, dtype=bool)] = np.asarray(p_a, dtype=np.float64)[~np.asarray(present, dtype=bool)]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise EvaluationError("Competition blend probabilities are invalid")
    return p


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if (y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise EvaluationError("Metric arrays must align and contain binary labels/valid probabilities")
    pred = p >= threshold
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    f0 = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    clipped = np.clip(p, LOGLOSS_EPS, 1-LOGLOSS_EPS)
    return {
        "n": int(len(y)), "class_0": int((y == 0).sum()), "class_1": int((y == 1).sum()),
        "confusion_matrix": [[tn, fp], [fn, tp]], "macro_f1": float((f0+f1)/2),
        "positive_precision": float(tp/(tp+fp)) if tp+fp else 0.0,
        "positive_recall": float(tp/(tp+fn)) if tp+fn else 0.0,
        "brier": float(np.mean((p-y)**2)),
        "logloss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
        "threshold": float(threshold), "logloss_epsilon": LOGLOSS_EPS,
    }


def metrics_missing_threshold(y: np.ndarray, probability: np.ndarray,
                              missing: np.ndarray, missing_p: np.ndarray,
                              threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8); p = np.asarray(probability, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool); missing_p = np.asarray(missing_p, dtype=np.float64)
    if p.shape != y.shape or missing.shape != y.shape or missing_p.shape != y.shape:
        raise EvaluationError("Post-selected threshold vectors are not aligned")
    pred = p >= 0.5
    pred[missing] = missing_p[missing] >= threshold
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    f0 = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.0
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    clipped = np.clip(p, LOGLOSS_EPS, 1-LOGLOSS_EPS)
    return {
        "n": int(len(y)), "class_0": int((y == 0).sum()), "class_1": int((y == 1).sum()),
        "confusion_matrix": [[tn, fp], [fn, tp]], "macro_f1": float((f0+f1)/2),
        "positive_precision": float(tp/(tp+fp)) if tp+fp else 0.0,
        "positive_recall": float(tp/(tp+fn)) if tp+fn else 0.0,
        "brier": float(np.mean((p-y)**2)),
        "logloss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
        "present_threshold": 0.5, "missing_threshold": float(threshold),
        "logloss_epsilon": LOGLOSS_EPS,
    }


def summarize_view(y: np.ndarray, pred: Mapping[str, np.ndarray], mask: np.ndarray,
                   origin: int, groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    present = pred[f"E{origin}_present"]
    p_a = pred[f"E{origin}_pA"]
    p_uniform = pred[f"E{origin}_pP_UNIFORM"]
    p_aligned = pred[f"E{origin}_pP_ALIGNED"]
    p_permuted = pred[f"E{origin}_pP_PERMUTED"]
    comp_base = pred[f"E{origin}_MIX50_REFERENCE"]
    comp_aligned = competition_probabilities(p_aligned, p_a, present)
    component_probs = {"ALIGNED": p_aligned, "PERMUTED": p_permuted, "UNIFORM": p_uniform}
    component_full = {k: metrics(y[mask], v[mask]) for k, v in component_probs.items()}
    component_present = {k: metrics(y[mask & present], v[mask & present])
                         for k, v in component_probs.items()}
    comp_full = {"MIX50_ORIGINAL": metrics(y[mask], comp_base[mask]),
                 "MIX50_ALIGNED": metrics(y[mask], comp_aligned[mask])}
    comp_present = {"MIX50_ORIGINAL": metrics(y[mask & present], comp_base[mask & present]),
                    "MIX50_ALIGNED": metrics(y[mask & present], comp_aligned[mask & present])}
    missing_local = ~present[mask]
    p_a_local = p_a[mask]
    operational = {
        "threshold": OPERATIONAL_MISSING_THRESHOLD,
        "interpretation": "post-selected descriptive operational reference; no tuning",
        "MIX50_ORIGINAL": metrics_missing_threshold(y[mask], comp_base[mask], missing_local,
                                                     p_a_local, OPERATIONAL_MISSING_THRESHOLD),
        "MIX50_ALIGNED": metrics_missing_threshold(y[mask], comp_aligned[mask], missing_local,
                                                   p_a_local, OPERATIONAL_MISSING_THRESHOLD),
    }
    subgroups: dict[str, Any] = {}
    for family in GROUPS:
        values = groups[family].astype(str)
        levels: dict[str, Any] = {}
        for level in sorted(set(values[mask].tolist())):
            gm = mask & (values == level); gy = y[gm]
            support = {"n": int(gm.sum()), "class_0": int((gy == 0).sum()),
                       "class_1": int((gy == 1).sum())}
            if (support["n"] < SUPPORT_N or support["class_0"] < SUPPORT_CLASS
                    or support["class_1"] < SUPPORT_CLASS):
                levels[level] = {"status": "UNKNOWN_SUPPORT", "support": support}
            else:
                base_m = metrics(gy, comp_base[gm]); candidate_m = metrics(gy, comp_aligned[gm])
                levels[level] = {
                    "status": "SUPPORTED", "support": support,
                    "MIX50_ORIGINAL": base_m, "MIX50_ALIGNED": candidate_m,
                    "harm_macro_f1": base_m["macro_f1"]-candidate_m["macro_f1"],
                    "harm_positive_recall": base_m["positive_recall"]-candidate_m["positive_recall"],
                }
        subgroups[family] = levels
    cycle_effects: dict[str, Any] = {}
    cycles = pred[f"E{origin}_cycles"]
    for cycle in sorted(np.unique(cycles[mask]).tolist()):
        cm = mask & (cycles == cycle)
        base_m = metrics(y[cm], comp_base[cm]); candidate_m = metrics(y[cm], comp_aligned[cm])
        gain = candidate_m["macro_f1"]-base_m["macro_f1"]
        cycle_effects[str(int(cycle))] = {
            "n": int(cm.sum()), "class_0": int((y[cm] == 0).sum()),
            "class_1": int((y[cm] == 1).sum()), "gain_macro_f1": gain,
            "direction": "POSITIVE" if gain > 0 else "NEGATIVE" if gain < 0 else "TIE",
        }
    return {
        "n": int(mask.sum()), "present_n": int((mask & present).sum()),
        "missing_n": int((mask & ~present).sum()),
        "component_full": component_full, "component_present_only": component_present,
        "competition_full": comp_full, "competition_present_only": comp_present,
        "competition_gains_macro_f1": {
            "full": comp_full["MIX50_ALIGNED"]["macro_f1"]-comp_full["MIX50_ORIGINAL"]["macro_f1"],
            "present_only": comp_present["MIX50_ALIGNED"]["macro_f1"]-comp_present["MIX50_ORIGINAL"]["macro_f1"],
        },
        "post_selected_missing_threshold_descriptive": operational,
        "subgroups": subgroups, "per_cycle": cycle_effects,
    }


def load_labels_after_manifest_commit(pred_manifest: Mapping[str, Any],
                                      pred_arrays: Mapping[str, np.ndarray]) -> dict[int, np.ndarray]:
    """The only function in this stage that reads query target values."""
    require_committed(REL_PREDICTION_MANIFEST)
    if (pred_manifest.get("status") != "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ"
            or pred_manifest.get("query_target_labels_read") is not False
            or pred_manifest.get("query_target_members_loaded") != []):
        raise EvaluationError("Prediction manifest does not prove a pre-label prediction freeze")
    if pred_manifest.get("prediction_sha256") != file_sha256(path(REL_PREDICTIONS)):
        raise EvaluationError("Saved predictions changed after manifest freeze")
    if git_text("rev-parse", f"HEAD:{REL_PREDICTION_MANIFEST}") != git_text("hash-object", REL_PREDICTION_MANIFEST):
        raise EvaluationError("Prediction manifest differs from committed Git blob")
    if file_sha256(path(TRAIN_REL)) != TRAIN_SHA256:
        raise EvaluationError("Official Train hash differs before target-label extraction")
    frame = pd.read_csv(path(TRAIN_REL), usecols=["record_id", "artifact_signal"],
                        dtype={"record_id": "string"})
    if len(frame) != 50608 or frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise EvaluationError("Official Train identity failed at target join")
    ids = frame.record_id.astype(str).to_numpy()
    labels: dict[int, np.ndarray] = {}
    for origin in (67, 75):
        rows = pred_arrays[f"E{origin}_rows"].astype(np.int64)
        expected_ids = pred_arrays[f"E{origin}_ids"].astype(str)
        if not np.array_equal(ids[rows], expected_ids):
            raise EvaluationError(f"E{origin} target rows do not join to frozen prediction IDs")
        y = pd.to_numeric(frame.iloc[rows]["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y, [0, 1]).all():
            raise EvaluationError(f"E{origin} evaluation labels are not binary")
        labels[origin] = y
    return labels


def evaluate() -> dict[str, Any]:
    freeze, freeze_commit = verify_source_freeze(require_frozen_commit=True)
    manifest_commit = require_committed(REL_PREDICTION_MANIFEST)
    pred_manifest = read_json(path(REL_PREDICTION_MANIFEST))
    if pred_manifest.get("source_freeze_sha256") != file_sha256(path(REL_FREEZE)):
        raise EvaluationError("Prediction manifest is not bound to the committed source freeze")
    if pred_manifest.get("prediction_sha256") != file_sha256(path(REL_PREDICTIONS)):
        raise EvaluationError("Prediction archive differs from the committed hash manifest")
    schema = pred_manifest.get("prediction_schema", [])
    with np.load(path(REL_PREDICTIONS), allow_pickle=False) as archive:
        arrays = read_allowlisted_members(archive, schema, schema)
    if any(key.endswith("_y") for key in arrays):
        raise EvaluationError("Prediction archive unexpectedly includes a target array")
    for key, expected_hash in pred_manifest.get("prediction_array_hashes", {}).items():
        if key not in arrays or sha256_array(arrays[key]) != expected_hash:
            raise EvaluationError(f"Saved prediction array hash mismatch: {key}")

    # First intentional target-value read occurs only after the committed
    # prediction manifest and archive have passed their hash/identity checks.
    labels = load_labels_after_manifest_commit(pred_manifest, arrays)
    views: dict[str, Any] = {}
    group_by_origin: dict[int, dict[str, np.ndarray]] = {}
    for origin in (67, 75):
        group_by_origin[origin] = {
            family: arrays[f"E{origin}_group_{family}"].astype(str) for family in GROUPS
        }
        cycles = arrays[f"E{origin}_cycles"].astype(np.int64)
        for name, (view_origin, first, last) in VIEWS.items():
            if origin != view_origin:
                continue
            mask = (cycles >= first) & (cycles <= last)
            views[name] = summarize_view(labels[origin], arrays, mask, origin, group_by_origin[origin])

    # Unique pool: E67 W4+W5 plus E75 W6. It excludes overlapping E67 W6.
    pieces: list[tuple[int, np.ndarray]] = []
    for origin, selected_views in ((67, ("E67_W4", "E67_W5")), (75, ("E75_W6",))):
        cycles = arrays[f"E{origin}_cycles"].astype(np.int64)
        selected = np.zeros(len(cycles), dtype=bool)
        for name in selected_views:
            _, first, last = VIEWS[name]
            selected |= (cycles >= first) & (cycles <= last)
        pieces.append((origin, selected))
    y_pool = np.concatenate([labels[origin][mask] for origin,mask in pieces])
    pool_pred: dict[str, np.ndarray] = {}
    for key in ("pA", "pP_UNIFORM", "pP_ALIGNED", "pP_PERMUTED", "MIX50_REFERENCE"):
        pool_pred[key] = np.concatenate([arrays[f"E{origin}_{key}"][mask] for origin,mask in pieces])
    pool_present = np.concatenate([arrays[f"E{origin}_present"][mask] for origin,mask in pieces])
    pool_cycles = np.concatenate([arrays[f"E{origin}_cycles"][mask] for origin,mask in pieces])
    pool_groups = {family: np.concatenate([
        group_by_origin[origin][family][mask] for origin,mask in pieces
    ]) for family in GROUPS}
    pool_prob = competition_probabilities(pool_pred["pP_ALIGNED"], pool_pred["pA"], pool_present)
    pool_arrays = {
        "E99_present": pool_present, "E99_cycles": pool_cycles,
        "E99_pA": pool_pred["pA"], "E99_pP_UNIFORM": pool_pred["pP_UNIFORM"],
        "E99_pP_ALIGNED": pool_pred["pP_ALIGNED"], "E99_pP_PERMUTED": pool_pred["pP_PERMUTED"],
        "E99_MIX50_REFERENCE": pool_pred["MIX50_REFERENCE"],
    }
    views["UNIQUE_E67_W4_W5_PLUS_E75_W6"] = summarize_view(
        y_pool, pool_arrays, np.ones(len(y_pool), dtype=bool), 99, pool_groups)

    mechanism_checks: dict[str, Any] = {}
    for view_name in ("E67_H12", "E75_W6"):
        m = views[view_name]["component_full"]
        aligned = m["ALIGNED"]["macro_f1"]
        gain_perm = aligned-m["PERMUTED"]["macro_f1"]
        gain_uniform = aligned-m["UNIFORM"]["macro_f1"]
        mechanism_checks[view_name] = {
            "aligned_vs_permuted_gain": gain_perm,
            "aligned_vs_uniform_gain": gain_uniform,
            "pass": gain_perm > 1e-12 and gain_uniform > 1e-12,
        }
    comp_gains = {name: views[name]["competition_gains_macro_f1"]["full"]
                  for name in ("E67_H12", "E75_W6", "UNIQUE_E67_W4_W5_PLUS_E75_W6")}
    e67_harms = {name: -views[name]["competition_gains_macro_f1"]["full"]
                 for name in ("E67_W4", "E67_W5", "E67_W6")}
    subgroup_failures: list[dict[str, Any]] = []
    subgroup_unknown: list[str] = []
    for view_name, view in views.items():
        for family, levels in view["subgroups"].items():
            for level, entry in levels.items():
                label = f"{view_name}/{family}/{level}"
                if entry["status"] != "SUPPORTED":
                    subgroup_unknown.append(label)
                    continue
                if (entry["harm_macro_f1"] > 0.01
                        or entry["harm_positive_recall"] > 0.02):
                    subgroup_failures.append({
                        "group": label, "harm_macro_f1": entry["harm_macro_f1"],
                        "harm_positive_recall": entry["harm_positive_recall"],
                    })
    competition_checks = {
        "E67_H12_gain_at_least_0.0003": comp_gains["E67_H12"] >= 0.0003,
        "E75_W6_gain_nonnegative": comp_gains["E75_W6"] >= 0.0,
        "unique_pool_gain_at_least_0.0003":
            comp_gains["UNIQUE_E67_W4_W5_PLUS_E75_W6"] >= 0.0003,
        "each_E67_window_harm_at_most_0.003":
            all(harm <= 0.003 for harm in e67_harms.values()),
        "supported_subgroups_within_guards": not subgroup_failures,
    }
    result = {
        "schema_version": 1, "status": "SCORED_DISCLOSED_DEVELOPMENT_EVALUATION",
        "source_freeze_commit": freeze_commit, "predictions_manifest_commit": manifest_commit,
        "source_freeze_sha256": file_sha256(path(REL_FREEZE)),
        "predictions_manifest_sha256": file_sha256(path(REL_PREDICTION_MANIFEST)),
        "prediction_sha256": file_sha256(path(REL_PREDICTIONS)),
        "incident_disclosure": {
            "incident_report": INCIDENT_REL,
            "prior_inventory_materialized_E67_y_and_E75_y_for_names_shapes_dtypes": True,
            "pre_exposure_cannot_be_claimed": True,
            "reused_development_targets_not_fresh_holdout": True,
        },
        "estimand_notes": {
            "mechanism": "Aligned/permuted/uniform positional component with same HGB missing fallback",
            "competition": "0.5*newPOS+0.5*pA on present; pA on missing; compared with MIX50 present and pA missing",
            "primary_missing_threshold": 0.5,
            "post_selected_operational_missing_threshold": OPERATIONAL_MISSING_THRESHOLD,
            "E67_E75_W6_overlap_is_history_sensitivity_not_independent_replication": True,
            "unique_pool_rows": int(len(y_pool)),
            "unique_pool_members": ["E67_W4", "E67_W5", "E75_W6"],
        },
        "metrics": {
            "present_and_primary_missing_threshold": 0.5, "logloss_epsilon": LOGLOSS_EPS,
            "subgroup_support": {"n_min": SUPPORT_N, "class_0_min": SUPPORT_CLASS,
                                 "class_1_min": SUPPORT_CLASS},
        },
        "views": views,
        "mechanism_gate": {"checks": mechanism_checks,
                           "pass": all(item["pass"] for item in mechanism_checks.values())},
        "competition_gate": {
            "checks": competition_checks, "gains": comp_gains, "E67_window_harms": e67_harms,
            "supported_subgroup_failures": subgroup_failures,
            "unsupported_subgroups_unknown": subgroup_unknown,
            "pass": all(competition_checks.values()),
        },
        "target_rows_read": int(sum(len(y) for y in labels.values())),
        "test_or_samplesubmission_accessed": False, "new_fits": 0,
    }
    write_json_once(path(REL_RESULTS), result)
    report = render_report(result)
    report_path = path(REL_REPORT)
    if report_path.exists() and report_path.read_text(encoding="utf-8") != report:
        raise EvaluationError("Refusing to overwrite an existing evaluation report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not report_path.exists():
        report_path.write_text(report, encoding="utf-8", newline="\n")
    return result


def render_report(result: Mapping[str, Any]) -> str:
    gains = result["competition_gate"]["gains"]
    lines = [
        "# CLS POS Frequency Ridge — fixed-policy evaluation R1", "",
        "## Disclosure", "",
        "This is a repeated development-data evaluation, not a fresh or untouched holdout. Before this evaluator was frozen, a prior inventory materialized E67_y and E75_y to print only names, shapes, and dtypes. The access-order deviation is recorded in EVALUATION_ACCESS_INCIDENT_01.json; pre-exposure cannot be undone. No metric or fit was made from that inventory. These results must not be described as outcome-blind.", "",
        "## Frozen comparison", "",
        "The component comparison uses ALIGNED, PERMUTED, and original uniform POS probabilities with the same cached HGB missing-stream fallback. The competition comparison uses 0.5 * new POS + 0.5 * same-origin pA on present-stream rows and pA on missing-stream rows. Primary missing classification threshold is 0.50. The 0.53230976129477 missing threshold is reported only as a post-selected operational description.", "",
        "## Competition Macro F1 gains", "", "| View | Gain vs original MIX50 with pA missing |", "|---|---:|",
    ]
    for name in ("E67_H12", "E75_W6", "UNIQUE_E67_W4_W5_PLUS_E75_W6"):
        lines.append(f"| {name} | {gains[name]:+.6f} |")
    lines += [
        "", "## Decision gates", "",
        f"- Mechanism gate: **{'PASS' if result['mechanism_gate']['pass'] else 'FAIL'}**.",
        f"- Competition gate: **{'PASS' if result['competition_gate']['pass'] else 'FAIL'}**.",
        "- The unique pool is E67 W4 + E67 W5 + E75 W6. E67 W6 and E75 W6 overlap and are history sensitivity, not independent replications.",
        "- Unsupported subgroup cells remain UNKNOWN; the JSON has denominators and supported-group effects.",
        "- No model fit, Test read, or Kaggle action was performed.", "",
        "Full confusion matrices, precision/recall, calibration metrics, present-only results, per-cycle directions, subgroup supports, and gate checks are in EVALUATION_RESULTS_R1.json.", "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-preflight", action="store_true")
    modes.add_argument("--write-source-freeze", action="store_true")
    modes.add_argument("--prepare-predictions", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.source_preflight:
        value = source_preflight()
    elif args.write_source_freeze:
        qa = run_synthetic_evaluator_qa()
        value = build_source_freeze(synthetic_qa=qa)
        write_json_once(path(REL_FREEZE), value)
    elif args.prepare_predictions:
        value = prepare_predictions()
    else:
        value = evaluate()
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
