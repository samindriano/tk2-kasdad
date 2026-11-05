"""Fixed-policy evaluation for the frozen one-sided positional-ridge pilot.

Prediction preparation is target-free. Query labels are read only by
--evaluate, after the prediction archive and manifest are committed and
verified. This evaluator has no Test, SampleSubmission, Kaggle, or fit path.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from scripts import evaluate_optimization_v3_cls_pos_frequency_ridge_v1 as OLD_EVAL
from scripts import optimization_v3_cls_pos_frequency_ridge_v1 as OLD_RUNNER
from scripts import optimization_v3_cls_pos_one_sided_ridge_v1 as RUNNER
from src.models.pos_frequency_ridge_v1 import sha256_array
from src.models.pos_one_sided_ridge_v1 import route_probabilities


CAMPAIGN_ID = "CLS_POS_ONE_SIDED_RIDGE_V1"
PREDICTION_REL = "experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/predictions.npz"
PREDICTION_MANIFEST_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/PREDICTIONS_MANIFEST.json"
RESULTS_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/EVALUATION_RESULTS.json"
REPORT_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/EVALUATION_REPORT.md"
ORIGINS = (67, 75)
VIEWS = {
    "E67_H12": (67, 68, 79),
    "E67_W4": (67, 68, 71),
    "E67_W5": (67, 72, 75),
    "E67_W6": (67, 76, 79),
    "E75_W6": (75, 76, 79),
}
GROUPS = ("availability", "entry_friction", "text_load")
SUPPORT_N = 200
SUPPORT_CLASS = 30
OPERATIONAL_MISSING_THRESHOLD = 0.53230976129477
LOGLOSS_EPS = 1e-15
PRIMARY_THRESHOLD = 0.50
CANDIDATES = ("UNIFORM", "ONE_SIDED", "GLOBAL_STRENGTH")
POSITION_KEYS = (
    "score_rows", "score_record_ids", "score_cycles", "present_mask",
    "prob_bag", "prob_offset_real", "prob_missing_hgb",
)
PILOT_KEYS = (
    "score_rows", "score_record_ids", "score_cycles", "present_mask",
    "prob_bag", "prob_missing_hgb", "prob_route_bag",
)


class OneSidedEvaluationError(RuntimeError):
    """Raised when a frozen prediction, label, metric, or resource boundary fails."""


def path(relative: str) -> Path:
    return ROOT / relative


def file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(file_path: Path) -> dict[str, Any]:
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OneSidedEvaluationError(f"Expected JSON object: {file_path}")
    return value


def write_json_once(file_path: Path, value: Mapping[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), indent=2, ensure_ascii=False, allow_nan=False,
                         sort_keys=True) + "\n"
    if file_path.exists():
        if file_path.read_text(encoding="utf-8") != encoded:
            raise OneSidedEvaluationError(f"Refusing to overwrite immutable result: {file_path}")
        return
    with file_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def write_npz_once(file_path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        raise OneSidedEvaluationError(f"Refusing to overwrite prediction archive: {file_path}")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    RUNNER.write_bytes_exclusive(file_path, buffer.getvalue())


def competition_probability(p_pos: np.ndarray, p_a: np.ndarray,
                           present: np.ndarray) -> np.ndarray:
    pos = np.asarray(p_pos, dtype=np.float64)
    pa = np.asarray(p_a, dtype=np.float64)
    mask = np.asarray(present, dtype=bool)
    if pos.shape != pa.shape or mask.shape != pa.shape:
        raise OneSidedEvaluationError("Competition blend inputs do not align")
    result = 0.5 * pos + 0.5 * pa
    result = result.astype(np.float64, copy=True)
    result[~mask] = pa[~mask]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise OneSidedEvaluationError("Competition probabilities are invalid")
    return result


def _load_gamma(origin: int, arm: str) -> tuple[np.ndarray, dict[str, Any]]:
    fit_id = f"E{origin}_{arm}"
    artifact = path(f"{RUNNER.RUN_ROOT}/units/{fit_id}.npz")
    receipt_path = path(f"{RUNNER.RUN_ROOT}/receipts/{fit_id}.json")
    receipt = read_json(receipt_path)
    if (receipt.get("campaign_id") != CAMPAIGN_ID or receipt.get("fit_id") != fit_id
            or receipt.get("origin") != origin or receipt.get("arm") != arm
            or receipt.get("artifact_sha256") != file_sha256(artifact)):
        raise OneSidedEvaluationError(f"{fit_id} receipt identity or artifact hash mismatch")
    root_pass = read_json(path(RUNNER.ROOT_SOURCE_PASS_REL))
    source_freeze = RUNNER.verify_source_freeze(require_frozen_commit=True)
    if (receipt.get("root_source_pass_sha256") != file_sha256(path(RUNNER.ROOT_SOURCE_PASS_REL))
            or receipt.get("source_freeze_sha256") != file_sha256(path(RUNNER.SOURCE_FREEZE_REL))
            or receipt.get("source_commit") != root_pass.get("source_commit")
            or receipt.get("source_hashes") != source_freeze.get("source_hashes")
            or receipt.get("input_hashes") != source_freeze.get("input_hashes")):
        raise OneSidedEvaluationError(f"{fit_id} source lineage mismatch")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma"}:
            raise OneSidedEvaluationError(f"{fit_id} gamma archive has unexpected members")
        gamma = np.asarray(archive["gamma"], dtype=np.float64)
    if gamma.ndim != 1 or not np.isfinite(gamma).all() or sha256_array(gamma) != receipt.get("gamma_sha256"):
        raise OneSidedEvaluationError(f"{fit_id} gamma content/shape mismatch")
    return gamma, {"fit_id": fit_id, "artifact_sha256": file_sha256(artifact),
                   "receipt_sha256": file_sha256(receipt_path), "gamma_sha256": sha256_array(gamma)}


def _verify_fit_ledger() -> dict[str, Any]:
    events = RUNNER.read_events(path(RUNNER.LEDGER_REL))
    start_ids = [e.get("fit_id") for e in events if e.get("event") == "FIT_STARTED"]
    saved_ids = [e.get("fit_id") for e in events if e.get("event") == "FIT_SAVED"]
    failed_ids = [e.get("fit_id") for e in events if e.get("event") == "FIT_FAILED"]
    if (start_ids != list(RUNNER.FIT_IDS) or saved_ids != list(RUNNER.FIT_IDS)
            or failed_ids or any(e.get("campaign_id") != CAMPAIGN_ID for e in events)):
        raise OneSidedEvaluationError("Exactly four ordered, saved, failure-free fits are required")
    for origin in ORIGINS:
        for arm in ARMS:
            _load_gamma(origin, arm)
    return {
        "campaign_id": CAMPAIGN_ID,
        "fit_ids_in_order": list(start_ids),
        "fit_starts": len(start_ids),
        "fit_saves": len(saved_ids),
        "fit_failures": 0,
        "cap": len(RUNNER.FIT_IDS),
        "cap_enforced": True,
        "all_gamma_receipts_validated": True,
    }


def _read_identity_and_cache(origin: int, identity: pd.DataFrame,
                             a02: Mapping[str, np.ndarray]) -> dict[str, Any]:
    expected = read_json(path(f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_E{origin}.json"))
    train_cycles = identity["cycle_code"].to_numpy(dtype=np.int64)
    train_ids = identity["record_id"].astype(str).to_numpy()
    rows = np.flatnonzero(train_cycles > origin)
    ids = train_ids[rows]
    cycles = train_cycles[rows]
    if len(rows) != int(RUNNER.read_json(path(RUNNER.PROTOCOL_REL))["origins"][str(origin)]["query_rows"]):
        raise OneSidedEvaluationError(f"E{origin} query row count changed")
    OLD_EVAL.validate_ordered_row_identity(
        a02[f"E{origin}_rows"].astype(np.int64), a02[f"E{origin}_ids"].astype(str),
        a02[f"E{origin}_cycles"].astype(np.int64), rows, ids, cycles,
    )
    pos = OLD_EVAL.safe_npz_load(
        path(f"{OLD_EVAL.POS_BAG_ROOT}/E{origin}/predictions_E{origin}.npz"),
        POSITION_KEYS, POSITION_KEYS,
    )
    pilot = OLD_EVAL.safe_npz_load(
        path(f"{OLD_EVAL.PILOT_ROOT}/E{origin}/predictions_E{origin}.npz"),
        PILOT_KEYS, PILOT_KEYS,
    )
    for name, bundle in (("POS_BAG", pos), ("PILOT", pilot)):
        OLD_EVAL.validate_ordered_row_identity(
            bundle["score_rows"].astype(np.int64), bundle["score_record_ids"].astype(str),
            bundle["score_cycles"].astype(np.int64), rows, ids, cycles,
        )
        if not np.array_equal(bundle["present_mask"].astype(bool), ~a02[f"E{origin}_missing"].astype(bool)):
            raise OneSidedEvaluationError(f"E{origin} {name} availability routing differs from A02")
    if OLD_EVAL.row_metadata(rows, ids, cycles) != expected["score_row_metadata"]:
        raise OneSidedEvaluationError(f"E{origin} query rows differ from frozen matrix manifest")
    return {"rows": rows, "ids": ids, "cycles": cycles, "pos": pos, "pilot": pilot,
            "manifest": expected}


def prepare_predictions() -> dict[str, Any]:
    """Replay all four frozen predictors and write target-free predictions + manifest."""
    with OLD_EVAL.SharedJobLock():
        initial = RUNNER.require_resource("saved_replay", "before query prediction preparation")
        freeze = RUNNER.verify_source_freeze(require_frozen_commit=True)
        root_pass = RUNNER.require_root_source_pass()
        fit_audit = _verify_fit_ledger()
        identity_report = RUNNER.check_official_identities()
        identity = RUNNER._read_identity_frame()
        a02 = OLD_EVAL.safe_npz_load(path(OLD_EVAL.A02_REL), OLD_EVAL.A02_KEYS,
                                     OLD_EVAL.A02_KEYS)
        if any(key.endswith("_y") for key in a02):
            raise OneSidedEvaluationError("A02 query-target keys were materialized before prediction freeze")
        outputs: dict[str, np.ndarray] = {}
        lineages: dict[str, Any] = {}
        resource_samples = [{"stage": "INITIAL_UNDER_SHARED_LOCK", **initial}]
        for origin in ORIGINS:
            before = RUNNER.require_actual_boundary("saved_replay", f"E{origin} before allocation")
            cache = _read_identity_and_cache(origin, identity, a02)
            rows, ids, cycles = cache["rows"], cache["ids"], cache["cycles"]
            pos, pilot, manifest = cache["pos"], cache["pilot"], cache["manifest"]
            present = ~a02[f"E{origin}_missing"].astype(bool)
            p_a = a02[f"E{origin}_pA"].astype(np.float64)
            missing_hgb = pos["prob_missing_hgb"].astype(np.float64)
            p_bag_saved = pilot["prob_bag"].astype(np.float64)
            score_base = OLD_EVAL.load_matrix(manifest, "score_base")
            score_position_aug = OLD_EVAL.load_matrix(manifest, "score_position")
            if (score_position_aug.shape[0] != len(rows)
                    or (score_position_aug[:, :score_base.shape[1]] != score_base).nnz):
                raise OneSidedEvaluationError(f"E{origin} frozen query matrix/base prefix mismatch")
            score_position = score_position_aug[:, score_base.shape[1]:].tocsr()
            accounting = read_json(path(OLD_EVAL.PILOT_ACCOUNTING_REL))
            fit_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
            bag_entry = next((x for x in accounting["fit_units"] if x.get("fit_id") == fit_id), None)
            if bag_entry is None:
                raise OneSidedEvaluationError(f"E{origin} fixed bag fit accounting is missing")
            bag_rel = f"{OLD_EVAL.PILOT_ROOT}/units/{fit_id}.joblib"
            bag_receipt_rel = f"{OLD_EVAL.PILOT_ROOT}/receipts/{fit_id}.json"
            bag_path = path(bag_rel)
            bag_receipt_path = path(bag_receipt_rel)
            if (file_sha256(bag_path) != bag_entry["artifact_sha256"]
                    or file_sha256(bag_receipt_path) != bag_entry["receipt_sha256"]):
                raise OneSidedEvaluationError(f"E{origin} fixed bag artifact/receipt changed")
            bag = joblib.load(bag_path).get("estimator")
            coef = np.asarray(bag.coef_, dtype=np.float64)
            intercept = np.asarray(bag.intercept_, dtype=np.float64).reshape(-1)
            if coef.shape != (1, score_base.shape[1]) or intercept.shape != (1,):
                raise OneSidedEvaluationError(f"E{origin} saved C2_B estimator width mismatch")
            offset = np.asarray(score_base @ coef[0], dtype=np.float64).reshape(-1) + float(intercept[0])
            if not np.isfinite(offset).all():
                raise OneSidedEvaluationError(f"E{origin} saved C2_B query logits are non-finite")
            p_bag_replay = expit(offset)
            if not np.allclose(p_bag_replay, p_bag_saved, rtol=0.0, atol=2e-12):
                raise OneSidedEvaluationError(f"E{origin} saved C2_B probability replay differs")
            old_uniform_rel = f"{OLD_EVAL.POS_BAG_ROOT}/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz"
            old_uniform_receipt_rel = f"{OLD_EVAL.POS_BAG_ROOT}/receipts/POS_BAG_OFFSET_V1_E{origin}_REAL.json"
            gamma_uniform, uniform_lineage = OLD_EVAL.gamma_from_artifact(old_uniform_rel, old_uniform_receipt_rel)
            p_uniform = route_probabilities(score_position, offset, gamma_uniform, present, missing_hgb)
            p_uniform_saved = a02[f"E{origin}_pP"].astype(np.float64)
            if not np.array_equal(p_uniform, p_uniform_saved):
                if not np.allclose(p_uniform, p_uniform_saved, rtol=0.0, atol=2e-10):
                    raise OneSidedEvaluationError(f"E{origin} uniform gamma does not reproduce the saved POS reference")
            if not np.array_equal(p_uniform[~present], missing_hgb[~present]):
                raise OneSidedEvaluationError(f"E{origin} uniform POS changed missing-stream fallback")
            route_by_arm: dict[str, np.ndarray] = {"UNIFORM": p_uniform}
            candidate_lineages: dict[str, Any] = {}
            for arm in ARMS:
                gamma, lineage = _load_gamma(origin, arm)
                if len(gamma) != score_position.shape[1]:
                    raise OneSidedEvaluationError(f"E{origin}_{arm} gamma width differs from query matrix")
                route_by_arm[arm] = route_probabilities(score_position, offset, gamma, present, missing_hgb)
                candidate_lineages[arm] = lineage
            reference = OLD_EVAL.current_primary_mix50_reference(p_uniform, p_a, ~present)
            if not np.array_equal(reference[~present], p_a[~present]):
                raise OneSidedEvaluationError(f"E{origin} current primary reference missing fallback is not pA")
            groups = {
                family: a02[f"E{origin}_group_{family}"].astype(str)
                for family in GROUPS
            }
            if not np.array_equal(groups["availability"], np.where(present, "present", "missing")):
                raise OneSidedEvaluationError(f"E{origin} availability subgroup differs from routed availability")
            if any(group.shape != rows.shape for group in groups.values()):
                raise OneSidedEvaluationError(f"E{origin} saved subgroup arrays are not row aligned")
            outputs.update({
                f"E{origin}_rows": rows.astype(np.int64),
                f"E{origin}_ids": ids.astype(str),
                f"E{origin}_cycles": cycles.astype(np.int16),
                f"E{origin}_present": present.astype(bool),
                f"E{origin}_pA": p_a,
                f"E{origin}_pPOS_UNIFORM": route_by_arm["UNIFORM"],
                f"E{origin}_pPOS_ONE_SIDED": route_by_arm["ONE_SIDED"],
                f"E{origin}_pPOS_GLOBAL_STRENGTH": route_by_arm["GLOBAL_STRENGTH"],
                f"E{origin}_pMISSING_HGB": missing_hgb,
                f"E{origin}_MIX50_UNIFORM_PRIMARY": reference,
                **{f"E{origin}_group_{family}": group for family, group in groups.items()},
            })
            lineages[str(origin)] = {
                "rows": int(len(rows)), "present_rows": int(present.sum()),
                "missing_rows": int((~present).sum()),
                "cycle_range": [int(cycles.min()), int(cycles.max())],
                "matrix_manifest_sha256": file_sha256(path(f"{OLD_EVAL.PILOT_ROOT}/MATRIX_MANIFEST_E{origin}.json")),
                "score_base_sha256": manifest["matrices"]["score_base"]["sha256"],
                "score_position_sha256": manifest["matrices"]["score_position"]["sha256"],
                "c2b_artifact_sha256": file_sha256(bag_path),
                "c2b_receipt_sha256": file_sha256(bag_receipt_path),
                "uniform_gamma": uniform_lineage,
                "new_gammas": candidate_lineages,
                "row_ids_cycles": "PASS_EXACT_ORDERED_ID_AND_CYCLE_ALIGNMENT",
                "c2b_probability_replay_max_abs_error": float(np.max(np.abs(p_bag_replay - p_bag_saved))),
                "uniform_saved_pos_replay_max_abs_error": float(np.max(np.abs(p_uniform - p_uniform_saved))),
                "missing_fallback": "BIT_EXACT_SAME_ORIGIN_HGB_FOR_POS; pA_EXACT_FOR_PRIMARY_COMPETITION_REFERENCE",
            }
            boundary = RUNNER.require_actual_boundary("saved_replay", f"E{origin} after matrix/prediction work")
            resource_samples.extend([{"stage": f"E{origin}_BEFORE", **before},
                                     {"stage": f"E{origin}_BOUNDARY", **boundary}])
            del score_base, score_position_aug, score_position, bag, coef, intercept, offset
        final_resource = RUNNER.require_actual_boundary("saved_replay", "after all origins before prediction save")
        resource_samples.append({"stage": "FINAL_BEFORE_SAVE", **final_resource})
        prediction_path = path(PREDICTION_REL)
        write_npz_once(prediction_path, outputs)
        manifest = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "status": "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ",
            "source_commit": freeze["source_commit"],
            "source_freeze_sha256": file_sha256(path(RUNNER.SOURCE_FREEZE_REL)),
            "source_hashes": freeze["source_hashes"],
            "input_hashes": freeze["input_hashes"],
            "root_source_pass_sha256": file_sha256(path(RUNNER.ROOT_SOURCE_PASS_REL)),
            "prediction_path": PREDICTION_REL,
            "prediction_sha256": file_sha256(prediction_path),
            "prediction_array_hashes": {key: sha256_array(value) for key, value in outputs.items()},
            "prediction_schema": list(outputs),
            "lineages": lineages,
            "fit_ledger_audit": fit_audit,
            "identity_preflight_sha256": file_sha256(path(RUNNER.SOURCE_PREFLIGHT_REL)),
            "target_npz_members_requested": [],
            "query_target_labels_read": False,
            "test_or_samplesubmission_accessed": False,
            "resource_snapshot": {
                "policy": RUNNER.RESOURCE_POLICIES["saved_replay"],
                "samples": resource_samples,
                "samples_are_peak_measurements": False,
            },
            "estimator_fits": 0,
        }
        write_json_once(path(PREDICTION_MANIFEST_REL), manifest)
        return manifest


def read_query_labels_after_manifest_commit(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], *,
    repo_root: Path = ROOT,
    train_path: Path | None = None,
    prediction_path: Path | None = None,
    manifest_rel: str = PREDICTION_MANIFEST_REL,
    expected_train_sha256: str | None = RUNNER.TRAIN_SHA256,
    expected_train_rows: int | None = 50608,
    expected_cycle_bounds: tuple[int, int] | None = (47, 79),
) -> dict[int, np.ndarray]:
    """Only reads query targets after all frozen predictions are committed and checked."""
    def git_local(*args: str) -> str:
        proc = __import__("subprocess").run(["git", *args], cwd=repo_root, text=True,
                                           capture_output=True, check=False)
        if proc.returncode:
            raise OneSidedEvaluationError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout.strip()

    if git_local("rev-parse", f"HEAD:{manifest_rel}") != git_local("hash-object", manifest_rel):
        raise OneSidedEvaluationError("Prediction manifest must be committed before query-label access")
    if (manifest.get("status") != "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ"
            or manifest.get("query_target_labels_read") is not False
            or manifest.get("target_npz_members_requested") != []):
        raise OneSidedEvaluationError("Prediction manifest does not prove a target-free prediction boundary")
    prediction_path = path(PREDICTION_REL) if prediction_path is None else prediction_path
    if file_sha256(prediction_path) != manifest.get("prediction_sha256"):
        raise OneSidedEvaluationError("Prediction archive hash changed after manifest freeze")
    train_path = path(TRAIN_REL) if train_path is None else train_path
    if expected_train_sha256 is not None and file_sha256(train_path) != expected_train_sha256:
        raise OneSidedEvaluationError("Official Train byte hash changed before label extraction")
    identity = RUNNER.read_identity_frame(train_path, expected_train_sha256,
                                          expected_train_rows, expected_cycle_bounds)
    wanted: dict[int, str] = {}
    for origin in ORIGINS:
        rows = np.asarray(arrays[f"E{origin}_rows"], dtype=np.int64)
        ids = np.asarray(arrays[f"E{origin}_ids"]).astype(str)
        cycles = np.asarray(arrays[f"E{origin}_cycles"], dtype=np.int64)
        if (rows.shape != ids.shape or rows.shape != cycles.shape
                or len(np.unique(rows)) != len(rows)
                or not np.array_equal(identity.iloc[rows]["record_id"].astype(str).to_numpy(), ids)
                or not np.array_equal(identity.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64), cycles)):
            raise OneSidedEvaluationError(f"E{origin} committed prediction ID/cycle identity mismatch")
        for row, record_id in zip(rows.tolist(), ids.tolist()):
            if row in wanted and wanted[row] != record_id:
                raise OneSidedEvaluationError("Overlapping origin query IDs conflict")
            wanted[row] = record_id
    with train_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or "record_id" not in header or "artifact_signal" not in header:
            raise OneSidedEvaluationError("Official Train header lacks required query label columns")
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        labels_by_row: dict[int, int] = {}
        for row_index, record in enumerate(reader):
            if row_index not in wanted:
                continue
            if max(id_col, target_col) >= len(record) or record[id_col] != wanted[row_index]:
                raise OneSidedEvaluationError("Query label row does not join to committed prediction ID")
            try:
                value = int(record[target_col])
            except (TypeError, ValueError) as exc:
                raise OneSidedEvaluationError("Query target is not an integer binary label") from exc
            if value not in (0, 1):
                raise OneSidedEvaluationError("Query target is not binary")
            labels_by_row[row_index] = value
    if set(labels_by_row) != set(wanted):
        raise OneSidedEvaluationError("Some committed query prediction rows lack a joined target")
    return {
        origin: np.asarray([labels_by_row[int(row)] for row in arrays[f"E{origin}_rows"]], dtype=np.int8)
        for origin in ORIGINS
    }


def _competition_by_arm(arrays: Mapping[str, np.ndarray], origin: int) -> dict[str, np.ndarray]:
    present = np.asarray(arrays[f"E{origin}_present"], dtype=bool)
    p_a = np.asarray(arrays[f"E{origin}_pA"], dtype=np.float64)
    result = {}
    for candidate in CANDIDATES:
        pos = np.asarray(arrays[f"E{origin}_pPOS_{candidate}"], dtype=np.float64)
        result[candidate] = competition_probability(pos, p_a, present)
    return result


def _metric_map(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return OLD_EVAL.metrics(np.asarray(y, dtype=np.int8), np.asarray(p, dtype=np.float64),
                            threshold=PRIMARY_THRESHOLD)


def summarize_view(y: np.ndarray, arrays: Mapping[str, np.ndarray], mask: np.ndarray,
                   origin: int, groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    present = np.asarray(arrays[f"E{origin}_present"], dtype=bool)
    component = {
        candidate: np.asarray(arrays[f"E{origin}_pPOS_{candidate}"], dtype=np.float64)
        for candidate in CANDIDATES
    }
    competition = _competition_by_arm(arrays, origin)
    full = {candidate: _metric_map(y[mask], probability[mask])
            for candidate, probability in component.items()}
    present_only = {candidate: _metric_map(y[mask & present], probability[mask & present])
                    for candidate, probability in component.items()}
    competition_full = {candidate: _metric_map(y[mask], probability[mask])
                        for candidate, probability in competition.items()}
    competition_present = {candidate: _metric_map(y[mask & present], probability[mask & present])
                           for candidate, probability in competition.items()}
    missing_mask = ~present[mask]
    p_a_local = np.asarray(arrays[f"E{origin}_pA"], dtype=np.float64)[mask]
    operational = {
        candidate: OLD_EVAL.metrics_missing_threshold(
            y[mask], competition[candidate][mask], missing_mask, p_a_local,
            OPERATIONAL_MISSING_THRESHOLD,
        )
        for candidate in CANDIDATES
    }
    subgroup_results: dict[str, Any] = {}
    for family in GROUPS:
        values = np.asarray(groups[family]).astype(str)
        levels: dict[str, Any] = {}
        for level in sorted(set(values[mask].tolist())):
            gm = mask & (values == level)
            gy = y[gm]
            n0 = int(np.count_nonzero(gy == 0))
            n1 = int(np.count_nonzero(gy == 1))
            support = {"n": int(gm.sum()), "class_0": n0, "class_1": n1}
            if support["n"] < SUPPORT_N or n0 < SUPPORT_CLASS or n1 < SUPPORT_CLASS:
                levels[level] = {"status": "UNKNOWN_SUPPORT", "support": support}
                continue
            ref = _metric_map(gy, competition["UNIFORM"][gm])
            levels[level] = {
                "status": "SUPPORTED", "support": support,
                "reference_uniform": ref,
                "candidates": {
                    candidate: _metric_map(gy, competition[candidate][gm])
                    for candidate in ("ONE_SIDED", "GLOBAL_STRENGTH")
                },
                "harm_macro_f1": {
                    candidate: ref["macro_f1"] - _metric_map(gy, competition[candidate][gm])["macro_f1"]
                    for candidate in ("ONE_SIDED", "GLOBAL_STRENGTH")
                },
                "harm_positive_recall": {
                    candidate: ref["positive_recall"] - _metric_map(gy, competition[candidate][gm])["positive_recall"]
                    for candidate in ("ONE_SIDED", "GLOBAL_STRENGTH")
                },
            }
        subgroup_results[family] = levels
    cycles = np.asarray(arrays[f"E{origin}_cycles"], dtype=np.int64)
    per_cycle = {}
    for cycle in sorted(np.unique(cycles[mask]).tolist()):
        cm = mask & (cycles == cycle)
        y_cycle = y[cm]
        reference = _metric_map(y_cycle, competition["UNIFORM"][cm])
        by_candidate = {
            candidate: _metric_map(y_cycle, competition[candidate][cm])
            for candidate in ("ONE_SIDED", "GLOBAL_STRENGTH")
        }
        per_cycle[str(int(cycle))] = {
            "n": int(cm.sum()), "class_0": int(np.count_nonzero(y_cycle == 0)),
            "class_1": int(np.count_nonzero(y_cycle == 1)),
            "gain_macro_f1_vs_uniform": {
                candidate: by_candidate[candidate]["macro_f1"] - reference["macro_f1"]
                for candidate in by_candidate
            },
            "direction_vs_uniform": {
                candidate: ("POSITIVE" if by_candidate[candidate]["macro_f1"] > reference["macro_f1"]
                            else "NEGATIVE" if by_candidate[candidate]["macro_f1"] < reference["macro_f1"]
                            else "TIE")
                for candidate in by_candidate
            },
        }
    return {
        "rows": int(mask.sum()),
        "present_rows": int((mask & present).sum()),
        "missing_rows": int((mask & ~present).sum()),
        "component_full": full,
        "component_present_only": present_only,
        "competition_full": competition_full,
        "competition_present_only": competition_present,
        "competition_gain_macro_f1_vs_uniform": {
            candidate: competition_full[candidate]["macro_f1"] - competition_full["UNIFORM"]["macro_f1"]
            for candidate in CANDIDATES if candidate != "UNIFORM"
        },
        "competition_present_gain_macro_f1_vs_uniform": {
            candidate: competition_present[candidate]["macro_f1"] - competition_present["UNIFORM"]["macro_f1"]
            for candidate in CANDIDATES if candidate != "UNIFORM"
        },
        "post_selected_missing_threshold_descriptive": operational,
        "subgroups": subgroup_results,
        "per_cycle": per_cycle,
    }


def evaluate_results() -> dict[str, Any]:
    with OLD_EVAL.SharedJobLock():
        resource = RUNNER.require_resource("saved_replay", "before query target evaluation")
        freeze = RUNNER.verify_source_freeze(require_frozen_commit=True)
        pred_manifest = read_json(path(PREDICTION_MANIFEST_REL))
        if RUNNER.git_text("rev-parse", f"HEAD:{PREDICTION_MANIFEST_REL}") != RUNNER.git_text("hash-object", PREDICTION_MANIFEST_REL):
            raise OneSidedEvaluationError("Prediction manifest must be committed before query target read")
        if (pred_manifest.get("source_freeze_sha256") != file_sha256(path(RUNNER.SOURCE_FREEZE_REL))
                or pred_manifest.get("source_hashes") != freeze.get("source_hashes")
                or pred_manifest.get("input_hashes") != freeze.get("input_hashes")
                or pred_manifest.get("root_source_pass_sha256") != file_sha256(path(RUNNER.ROOT_SOURCE_PASS_REL))):
            raise OneSidedEvaluationError("Prediction manifest does not match committed source inputs")
        pred_path = path(PREDICTION_REL)
        if file_sha256(pred_path) != pred_manifest.get("prediction_sha256"):
            raise OneSidedEvaluationError("Prediction archive differs from committed manifest")
        with np.load(pred_path, allow_pickle=False) as archive:
            schema = list(pred_manifest["prediction_schema"])
            if len(schema) != len(set(schema)) or set(schema) != set(archive.files):
                raise OneSidedEvaluationError("Prediction archive schema differs from manifest")
            arrays = {key: np.asarray(archive[key]) for key in schema}
        for key, expected in pred_manifest.get("prediction_array_hashes", {}).items():
            if key not in arrays or sha256_array(arrays[key]) != expected:
                raise OneSidedEvaluationError(f"Prediction array hash mismatch: {key}")
        if any(key.endswith("_y") for key in arrays):
            raise OneSidedEvaluationError("Prediction archive must not contain target arrays")
        labels = read_query_labels_after_manifest_commit(pred_manifest, arrays)

        views: dict[str, Any] = {}
        group_by_origin = {
            origin: {family: arrays[f"E{origin}_group_{family}"].astype(str) for family in GROUPS}
            for origin in ORIGINS
        }
        for name, (origin, first, last) in VIEWS.items():
            cycles = arrays[f"E{origin}_cycles"].astype(np.int64)
            mask = (cycles >= first) & (cycles <= last)
            views[name] = summarize_view(labels[origin], arrays, mask, origin, group_by_origin[origin])

        pool_pieces: list[tuple[int, np.ndarray]] = []
        for origin, names in ((67, ("E67_W4", "E67_W5")), (75, ("E75_W6",))):
            cycles = arrays[f"E{origin}_cycles"].astype(np.int64)
            select = np.zeros(len(cycles), dtype=bool)
            for name in names:
                _, first, last = VIEWS[name]
                select |= (cycles >= first) & (cycles <= last)
            pool_pieces.append((origin, select))
        y_pool = np.concatenate([labels[origin][mask] for origin, mask in pool_pieces])
        pool_arrays: dict[str, np.ndarray] = {}
        for arm in CANDIDATES:
            key = f"pPOS_{arm}"
            pool_arrays[f"E99_{key}"] = np.concatenate([
                arrays[f"E{origin}_{key}"][mask] for origin, mask in pool_pieces
            ])
        for key in ("present", "pA", "cycles"):
            pool_arrays[f"E99_{key}"] = np.concatenate([
                arrays[f"E{origin}_{key}"][mask] for origin, mask in pool_pieces
            ])
        pool_groups = {family: np.concatenate([
            group_by_origin[origin][family][mask] for origin, mask in pool_pieces
        ]) for family in GROUPS}
        views["UNIQUE_E67_W4_W5_PLUS_E75_W6"] = summarize_view(
            y_pool, pool_arrays, np.ones(len(y_pool), dtype=bool), 99, pool_groups)

        mechanism_checks: dict[str, Any] = {}
        for view_name in ("E67_H12", "E75_W6"):
            component = views[view_name]["component_full"]
            f1_one = component["ONE_SIDED"]["macro_f1"]
            gain_uniform = f1_one - component["UNIFORM"]["macro_f1"]
            gain_global = f1_one - component["GLOBAL_STRENGTH"]["macro_f1"]
            mechanism_checks[view_name] = {
                "one_sided_vs_uniform_macro_f1": gain_uniform,
                "one_sided_vs_global_strength_macro_f1": gain_global,
                "minimum_margin": 1e-12,
                "pass": gain_uniform > 1e-12 and gain_global > 1e-12,
            }
        gains = {
            name: views[name]["competition_gain_macro_f1_vs_uniform"]["ONE_SIDED"]
            for name in ("E67_H12", "E75_W6", "UNIQUE_E67_W4_W5_PLUS_E75_W6")
        }
        window_harms = {
            name: -views[name]["competition_gain_macro_f1_vs_uniform"]["ONE_SIDED"]
            for name in ("E67_W4", "E67_W5", "E67_W6")
        }
        subgroup_failures: list[dict[str, Any]] = []
        subgroup_unknown: list[str] = []
        for view_name, view in views.items():
            for family, levels in view["subgroups"].items():
                for level, entry in levels.items():
                    label = f"{view_name}/{family}/{level}"
                    if entry["status"] != "SUPPORTED":
                        subgroup_unknown.append(label)
                        continue
                    if (entry["harm_macro_f1"]["ONE_SIDED"] > 0.01
                            or entry["harm_positive_recall"]["ONE_SIDED"] > 0.02):
                        subgroup_failures.append({
                            "group": label,
                            "harm_macro_f1": entry["harm_macro_f1"]["ONE_SIDED"],
                            "harm_positive_recall": entry["harm_positive_recall"]["ONE_SIDED"],
                        })
        competition_checks = {
            "E67_H12_gain_at_least_0.0003": gains["E67_H12"] >= 0.0003,
            "E75_W6_gain_nonnegative": gains["E75_W6"] >= 0.0,
            "unique_pool_gain_at_least_0.0003": gains["UNIQUE_E67_W4_W5_PLUS_E75_W6"] >= 0.0003,
            "each_E67_window_harm_at_most_0.003": all(x <= 0.003 for x in window_harms.values()),
            "supported_subgroups_within_guards": not subgroup_failures,
        }
        final_resource = RUNNER.require_actual_boundary("saved_replay", "before evaluation result save")
        result = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "status": "SCORED_REPEATED_DEVELOPMENT_EVALUATION",
            "source_freeze_sha256": file_sha256(path(RUNNER.SOURCE_FREEZE_REL)),
            "predictions_manifest_sha256": file_sha256(path(PREDICTION_MANIFEST_REL)),
            "prediction_sha256": file_sha256(pred_path),
            "resource_snapshot": {
                "policy": RUNNER.RESOURCE_POLICIES["saved_replay"],
                "initial_under_shared_lock": resource,
                "actual_boundary_before_save": final_resource,
                "samples_are_peak_measurements": False,
            },
            "historical_evidence": {
                "parent_campaign": "CLS_POS_FREQUENCY_RIDGE_V1",
                "adaptively_selected_after_failed_R1_gates": True,
                "causal_claim": False,
                "reused_development_targets_are_fresh_holdout": False,
                "prior_aligned_vs_uniform_negative_results": read_json(path(RUNNER.SOURCE_PREFLIGHT_REL))["identity_checks"]["prior_mean_one_negative_evidence"],
                "access_incident": OLD_EVAL.INCIDENT_REL,
            },
            "estimand": {
                "component": "same-origin frozen bag-logit offset plus positional gamma; HGB fallback retained on stream-missing rows",
                "competition": "present: 0.5*candidate_POS + 0.5*pA; missing: pA exactly; threshold 0.50",
                "primary_reference": "saved UNIFORM POS with same pA blend and pA missing fallback",
                "historical_A02_MIX50_missing_fallback": "pP; descriptive archive only, not the primary comparator",
                "post_selected_missing_threshold": OPERATIONAL_MISSING_THRESHOLD,
                "overlap": "E67 H12 contains E67 W6; E67 W6 and E75 W6 overlap in cycles 76-79 and are not independent replications",
                "unique_pool_members": ["E67_W4", "E67_W5", "E75_W6"],
                "unique_pool_n": int(len(y_pool)),
            },
            "metrics": {
                "primary_threshold": PRIMARY_THRESHOLD,
                "post_selected_threshold_is_descriptive_only": True,
                "logloss_epsilon": LOGLOSS_EPS,
                "subgroup_support": {"n_min": SUPPORT_N, "class_0_min": SUPPORT_CLASS,
                                     "class_1_min": SUPPORT_CLASS},
            },
            "views": views,
            "mechanism_gate": {"checks": mechanism_checks,
                               "pass": all(item["pass"] for item in mechanism_checks.values())},
            "competition_gate": {
                "checks": competition_checks,
                "gains_vs_saved_uniform_reference": gains,
                "E67_window_harms": window_harms,
                "supported_subgroup_failures": subgroup_failures,
                "unsupported_subgroups_unknown": subgroup_unknown,
                "pass": all(competition_checks.values()),
            },
            "target_rows_read": int(len(set(np.concatenate([arrays[f"E{o}_rows"] for o in ORIGINS]).tolist()))),
            "query_target_values_read_after_manifest_commit": True,
            "new_fits": 0,
            "test_or_samplesubmission_accessed": False,
            "kaggle_accessed": False,
        }
        write_json_once(path(RESULTS_REL), result)
        write_report_once(path(REPORT_REL), result)
        return result


def write_report_once(report_path: Path, result: Mapping[str, Any]) -> None:
    mechanism = result["mechanism_gate"]
    competition = result["competition_gate"]
    gains = competition["gains_vs_saved_uniform_reference"]
    lines = [
        "# CLS one-sided positional ridge — fixed-policy evaluation", "",
        "## Disclosure", "",
        "This is a repeated development-data evaluation. The origins and query cycles were used in earlier model development, including the mean-one ALIGNED test whose gate failures motivated this adaptive follow-up. Results are not independent confirmation and do not support causal interpretation. The historical access incident remains recorded in the inherited research report.", "",
        "## Frozen comparison", "",
        "The one-sided map is exactly `v_j = max(1, w_j)` from the saved prefix-only ALIGNED map, with no renormalization. The GLOBAL_STRENGTH control uses the single scalar `mean(v)` repeated for every coordinate. Saved UNIFORM coefficients and predictions are reused. Both new arms use the same frozen C2_B offset and exact missing-stream HGB fallback.", "",
        "## Main results", "",
        "| View | One-sided component Macro F1 gain vs uniform | One-sided vs global component Macro F1 | Competition gain vs same-policy uniform reference |", "|---|---:|---:|---:|",
    ]
    for view in ("E67_H12", "E75_W6", "UNIQUE_E67_W4_W5_PLUS_E75_W6"):
        item = result["views"][view]
        component = item["component_full"]
        comp_one = component["ONE_SIDED"]["macro_f1"]
        comp_uniform = component["UNIFORM"]["macro_f1"]
        comp_global = component["GLOBAL_STRENGTH"]["macro_f1"]
        competition_gain = item["competition_gain_macro_f1_vs_uniform"].get("ONE_SIDED")
        if competition_gain is None:
            competition_gain = item["competition_gain_macro_f1_vs_uniform"]["ONE_SIDED"]
        lines.append(f"| {view} | {comp_one-comp_uniform:+.6f} | {comp_one-comp_global:+.6f} | {competition_gain:+.6f} |")
    lines += [
        "", f"- Mechanism gate: **{'PASS' if mechanism['pass'] else 'FAIL'}**.",
        f"- Competition gate: **{'PASS' if competition['pass'] else 'FAIL'}**.",
        "- The primary competition reference uses pA on missing-stream rows. The archived A02 MIX50 missing fallback used pP and is not the primary comparator here.",
        "- E67 H12 contains E67 W6. E67 W6 and E75 W6 share cycles 76–79; they are history sensitivity, not independent replication.",
        "- The unique pool is E67 W4 + E67 W5 + E75 W6. Unsupported subgroup cells remain UNKNOWN.",
        "- The previous mean-one ALIGNED recipe's negative gates remain counterevidence. This new result, whatever its value, does not update the old verdict.",
        "- No Test data, Kaggle score, or submission was used.", "",
        "Full and present-only component/competition metrics, confusion matrices, precision/recall, calibration metrics, per-cycle signs, subgroup denominators, all negative findings, and UNKNOWN cells are in EVALUATION_RESULTS.json.", "",
    ]
    text = "\n".join(lines)
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != text:
            raise OneSidedEvaluationError(f"Refusing to overwrite evaluation report: {report_path}")
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-predictions", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_predictions() if args.prepare_predictions else evaluate_results()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        return 0
    except (OneSidedEvaluationError, RUNNER.OneSidedRunnerError, OLD_EVAL.EvaluationError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
