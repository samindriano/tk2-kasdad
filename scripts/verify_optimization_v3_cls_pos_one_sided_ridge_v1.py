"""Independent checks for CLS_POS_ONE_SIDED_RIDGE_V1 evidence.

The preflight mode is identity-only. Prediction checks use frozen artifacts but
no labels. Result verification opens query targets only after verifying that
the prediction manifest is committed and its archive hashes match.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_optimization_v3_cls_pos_frequency_ridge_v1 as OLD_EVAL
from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr


CAMPAIGN_ID = "CLS_POS_ONE_SIDED_RIDGE_V1"
PREFLIGHT_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/SOURCE_PREFLIGHT.json"
FREEZE_REL = "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/SOURCE_FREEZE.json"
ROOT_PASS_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/ROOT_SOURCE_PASS.json"
PRED_REL = "experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/predictions.npz"
PRED_MANIFEST_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/PREDICTIONS_MANIFEST.json"
RESULTS_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/EVALUATION_RESULTS.json"
TRAIN_REL = "data/raw/classification/Train.csv"
TRAIN_SHA = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
EXPECTED_SOURCES = {
    "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/protocol_r1.json",
    "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/protocol_r2.json",
    "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/PROTOCOL_R2_CORRECTION.md",
    "src/models/pos_one_sided_ridge_v1.py",
    "scripts/optimization_v3_cls_pos_one_sided_ridge_v1.py",
    "scripts/evaluate_optimization_v3_cls_pos_one_sided_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_one_sided_ridge_v1.py",
    "tests/test_pos_one_sided_ridge_v1.py",
}
ORIGINS = (67, 75)
ARMS = ("ONE_SIDED", "GLOBAL_STRENGTH")
FIT_IDS = tuple(f"E{origin}_{arm}" for origin in ORIGINS for arm in ARMS)
CANDIDATES = ("UNIFORM", "ONE_SIDED", "GLOBAL_STRENGTH")
VIEWS = {
    "E67_H12": (67, 68, 79), "E67_W4": (67, 68, 71),
    "E67_W5": (67, 72, 75), "E67_W6": (67, 76, 79),
    "E75_W6": (75, 76, 79),
}
GROUPS = ("availability", "entry_friction", "text_load")
EPS = 1e-15
TOL = 2e-12
PRIMARY_THRESHOLD = 0.5
OPERATIONAL_MISSING_THRESHOLD = 0.53230976129477
SUPPORT_N = 200
SUPPORT_CLASS = 30


class IndependentVerificationError(AssertionError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha(value: Any) -> str:
    array = np.asarray(value)
    header = canonical_sha({"shape": array.shape, "dtype": str(array.dtype)})
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def json_file(relative: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentVerificationError(f"Expected JSON object: {relative}")
    return value


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise IndependentVerificationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def row_metadata(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, Any]:
    selected = frame.iloc[np.asarray(rows, dtype=np.int64)]
    ids = selected["record_id"].astype(str).to_numpy()
    cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    return {
        "rows": int(len(rows)),
        "cycle_min": int(cycles.min()),
        "cycle_max": int(cycles.max()),
        "row_indices_sha256": array_sha(rows),
        "record_ids_sha256": array_sha(ids),
        "cycles_sha256": array_sha(cycles),
    }


def verify_source_preflight() -> dict[str, Any]:
    report = json_file(PREFLIGHT_REL)
    if (report.get("campaign_id") != CAMPAIGN_ID
            or report.get("status") != "PASS_TARGET_FREE_IDENTITY_PREFLIGHT"
            or report.get("target_values_opened") is not False
            or report.get("target_npz_members_requested") != []
            or report.get("official_fit_started") is not False
            or report.get("query_predictions_created") is not False
            or report.get("test_or_samplesubmission_accessed") is not False
            or report.get("kaggle_accessed") is not False):
        raise IndependentVerificationError("Source preflight status/boundary flags are not target-free")
    if not EXPECTED_SOURCES.issubset(report.get("source_hashes", {})):
        raise IndependentVerificationError("Source preflight omits one or more core source files")
    for relative, expected in {**report["source_hashes"], **report["input_hashes"]}.items():
        path = ROOT / relative
        if not path.is_file() or sha_file(path) != expected:
            raise IndependentVerificationError(f"Preflight source/input byte hash mismatch: {relative}")
        lower = relative.replace("\\", "/").lower()
        if lower.endswith("/test.csv") or "samplesubmission" in lower or lower.endswith("/submission.csv"):
            raise IndependentVerificationError(f"Forbidden Test/submission path in preflight: {relative}")
    train_path = ROOT / TRAIN_REL
    if sha_file(train_path) != TRAIN_SHA or report.get("identity_checks", {}).get("train_sha256") != TRAIN_SHA:
        raise IndependentVerificationError("Official Train byte hash differs from the frozen input")
    identity = pd.read_csv(train_path, usecols=["record_id", "cycle_code"],
                           dtype={"record_id": "string"})
    if len(identity) != 50608 or identity.record_id.isna().any() or identity.record_id.duplicated().any():
        raise IndependentVerificationError("Train ID/cycle identity schema is invalid")
    cycles = pd.to_numeric(identity["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (int(cycles.min()), int(cycles.max())) != (47, 79):
        raise IndependentVerificationError("Train cycle boundary is not the frozen 47-79 interval")
    checks = 0
    for origin, fit_rows_expected, query_rows_expected in ((67, 29705, 20903), (75, 42117, 8491)):
        fit_rows = np.flatnonzero(cycles <= origin)
        query_rows = np.flatnonzero(cycles > origin)
        if (len(fit_rows) != fit_rows_expected or len(query_rows) != query_rows_expected
                or row_metadata(identity, fit_rows) != report["identity_checks"]["origins"][str(origin)]["identity_and_cache"]["fit_row_metadata"]
                or row_metadata(identity, query_rows) != report["identity_checks"]["origins"][str(origin)]["identity_and_cache"]["score_row_metadata"]):
            raise IndependentVerificationError(f"E{origin} ordered row identities do not reproduce")
        identity_entry = report["identity_checks"]["origins"][str(origin)]
        checks_entry = identity_entry["identity_and_cache"]
        manifest_rel = f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_E{origin}.json"
        if sha_file(ROOT / manifest_rel) != checks_entry["matrix_manifest_sha256"]:
            raise IndependentVerificationError(f"E{origin} matrix manifest hash changed")
        manifest = json_file(manifest_rel)
        for matrix_name, expected_hash in checks_entry["matrix_file_hashes"].items():
            matrix_path = ROOT / manifest["matrices"][matrix_name]["path"]
            if sha_file(matrix_path) != expected_hash:
                raise IndependentVerificationError(f"E{origin} matrix hash mismatch: {matrix_name}")
        # Load identity fields by explicit allowlist only. No target NPZ field is named or expanded.
        cache_rel = f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/E{origin}/predictions_E{origin}.npz"
        with np.load(ROOT / cache_rel, allow_pickle=False) as archive:
            score_rows = np.asarray(archive["score_rows"], dtype=np.int64)
            score_ids = np.asarray(archive["score_record_ids"]).astype(str)
            score_cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
        if (not np.array_equal(score_rows, query_rows)
                or not np.array_equal(score_ids, identity.iloc[query_rows].record_id.astype(str).to_numpy())
                or not np.array_equal(score_cycles, cycles[query_rows])):
            raise IndependentVerificationError(f"E{origin} saved pilot IDs/cycles do not match Train")
        map_receipt_rel = f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.json"
        map_artifact_rel = f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.npz"
        map_receipt = json_file(map_receipt_rel)
        if (sha_file(ROOT / map_artifact_rel) != identity_entry["saved_aligned_map_artifact_sha256"]
                or sha_file(ROOT / map_receipt_rel) != identity_entry["saved_aligned_map_receipt_sha256"]
                or map_receipt.get("artifact_sha256") != sha_file(ROOT / map_artifact_rel)):
            raise IndependentVerificationError(f"E{origin} saved prefix-map hashes do not reproduce")
        checks += 7
    if report.get("resource_snapshot", {}).get("initial_under_shared_lock", {}).get("policy") != "source_preflight":
        raise IndependentVerificationError("Source preflight did not record the correct resource policy")
    negatives = report["identity_checks"]["prior_mean_one_negative_evidence"]
    if (negatives["E67_H12"]["one_sided_parent_aligned_vs_uniform"] >= 0
            or negatives["E75_W6"]["one_sided_parent_aligned_vs_uniform"] >= 0
            or negatives["UNIQUE_E67_W4_W5_PLUS_E75_W6"]["competition_gain_vs_primary"] >= 0):
        raise IndependentVerificationError("Prior negative gate counterevidence is missing or changed")
    return {"status": "PASS_INDEPENDENT_TARGET_FREE_IDENTITY_PREFLIGHT",
            "identity_matrix_map_checks": checks,
            "source_hash_count": len(report["source_hashes"]),
            "input_hash_count": len(report["input_hashes"]),
            "target_values_opened": False,
            "target_npz_members_loaded": [],
            "test_accessed": False}


def verify_source_freeze() -> dict[str, Any]:
    freeze = json_file(FREEZE_REL)
    body = dict(freeze)
    claimed = body.pop("freeze_signature", None)
    if claimed != canonical_sha(body):
        raise IndependentVerificationError("Source freeze canonical self-signature mismatch")
    if (freeze.get("campaign_id") != CAMPAIGN_ID
            or freeze.get("status") != "FROZEN_SOURCE_ONLY_NO_FIT_AUTHORIZATION"
            or freeze.get("official_fits_authorized") is not False):
        raise IndependentVerificationError("Source freeze does not preserve source-only state")
    for relative, expected in {**freeze["source_hashes"], **freeze["input_hashes"]}.items():
        if not (ROOT / relative).is_file() or sha_file(ROOT / relative) != expected:
            raise IndependentVerificationError(f"Source freeze file hash mismatch: {relative}")
    if sha_file(ROOT / PREFLIGHT_REL) != freeze.get("source_preflight_sha256"):
        raise IndependentVerificationError("Source freeze is not bound to the committed target-free preflight")
    if git("rev-parse", f"HEAD:{FREEZE_REL}") != git("hash-object", FREEZE_REL):
        raise IndependentVerificationError("Source freeze is not committed unchanged")
    return freeze


def verify_gamma(origin: int, arm: str, freeze: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    fit_id = f"E{origin}_{arm}"
    artifact_rel = f"experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/units/{fit_id}.npz"
    receipt_rel = f"experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/receipts/{fit_id}.json"
    artifact, receipt = ROOT / artifact_rel, json_file(receipt_rel)
    if (receipt.get("campaign_id") != CAMPAIGN_ID or receipt.get("fit_id") != fit_id
            or receipt.get("fit_kind") != "PREDICTOR"
            or receipt.get("source_hashes") != freeze.get("source_hashes")
            or receipt.get("input_hashes") != freeze.get("input_hashes")
            or receipt.get("artifact_sha256") != sha_file(artifact)):
        raise IndependentVerificationError(f"{fit_id} receipt/source/artifact lineage mismatch")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != {"gamma"}:
            raise IndependentVerificationError(f"{fit_id} gamma archive contains unexpected arrays")
        gamma = np.asarray(archive["gamma"], dtype=np.float64)
    if gamma.ndim != 1 or not np.isfinite(gamma).all() or array_sha(gamma) != receipt.get("gamma_sha256"):
        raise IndependentVerificationError(f"{fit_id} gamma logical hash/finiteness check failed")
    map_rel = f"experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/maps/penalty_map_E{origin}.npz"
    with np.load(ROOT / map_rel, allow_pickle=False) as archive:
        aligned = np.asarray(archive["aligned_token_weights"], dtype=np.float64)
    v = np.maximum(1.0, aligned)
    c = float(v.mean())
    expected = np.tile(v, 3) if arm == "ONE_SIDED" else np.full(3 * len(v), c, dtype=np.float64)
    if (receipt.get("penalty_vector_sha256") != array_sha(expected)
            or receipt.get("global_strength_scalar") != c
            or receipt.get("one_sided_renormalized") is True):
        raise IndependentVerificationError(f"{fit_id} penalty vector does not match frozen recipe")
    return gamma, {"fit_id": fit_id, "artifact_sha256": sha_file(artifact),
                   "receipt_sha256": sha_file(ROOT / receipt_rel), "gamma_sha256": array_sha(gamma),
                   "penalty_sha256": array_sha(expected), "global_strength_scalar": c}


def verify_prediction_bundle() -> dict[str, Any]:
    freeze = verify_source_freeze()
    pred_manifest = json_file(PRED_MANIFEST_REL)
    pred_path = ROOT / PRED_REL
    if (git("rev-parse", f"HEAD:{PRED_MANIFEST_REL}") != git("hash-object", PRED_MANIFEST_REL)
            or pred_manifest.get("status") != "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ"
            or pred_manifest.get("query_target_labels_read") is not False
            or pred_manifest.get("target_npz_members_requested") != []
            or pred_manifest.get("prediction_sha256") != sha_file(pred_path)
            or pred_manifest.get("source_freeze_sha256") != sha_file(ROOT / FREEZE_REL)):
        raise IndependentVerificationError("Committed prediction manifest does not bind the target-free bundle")
    with np.load(pred_path, allow_pickle=False) as archive:
        schema = pred_manifest["prediction_schema"]
        if set(schema) != set(archive.files):
            raise IndependentVerificationError("Prediction archive member names differ from manifest")
        arrays = {key: np.asarray(archive[key]) for key in schema}
    max_diffs: list[float] = []
    for key, expected in pred_manifest["prediction_array_hashes"].items():
        if key not in arrays or array_sha(arrays[key]) != expected:
            raise IndependentVerificationError(f"Prediction logical hash mismatch: {key}")
    checks = 0
    source_origins = {int(x) for x in ORIGINS}
    for origin in ORIGINS:
        manifest_rel = f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_E{origin}.json"
        manifest = json_file(manifest_rel)
        score_base = sparse.load_npz(ROOT / manifest["matrices"]["score_base"]["path"]).tocsr()
        augmented = sparse.load_npz(ROOT / manifest["matrices"]["score_position"]["path"]).tocsr()
        x = augmented[:, score_base.shape[1]:].tocsr()
        query_rows = arrays[f"E{origin}_rows"].astype(np.int64)
        query_ids = arrays[f"E{origin}_ids"].astype(str)
        query_cycles = arrays[f"E{origin}_cycles"].astype(np.int64)
        if (len(query_rows) != len(query_ids) or len(query_rows) != len(query_cycles)
                or len(np.unique(query_rows)) != len(query_rows)
                or x.shape[0] != len(query_rows)):
            raise IndependentVerificationError(f"E{origin} prediction row/matrix dimensions do not align")
        a02 = OLD_EVAL.safe_npz_load(ROOT / OLD_EVAL.A02_REL, OLD_EVAL.A02_KEYS, OLD_EVAL.A02_KEYS)
        OLD_EVAL.validate_ordered_row_identity(
            a02[f"E{origin}_rows"].astype(np.int64), a02[f"E{origin}_ids"].astype(str),
            a02[f"E{origin}_cycles"].astype(np.int64), query_rows, query_ids, query_cycles,
        )
        missing = a02[f"E{origin}_missing"].astype(bool)
        present = ~missing
        p_a = a02[f"E{origin}_pA"].astype(np.float64)
        if not np.array_equal(arrays[f"E{origin}_present"].astype(bool), present):
            raise IndependentVerificationError(f"E{origin} availability route changed")
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        accounting = json_file("reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json")
        bag_entry = next(item for item in accounting["fit_units"] if item.get("fit_id") == bag_id)
        bag_path = ROOT / f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/units/{bag_id}.joblib"
        if sha_file(bag_path) != bag_entry["artifact_sha256"]:
            raise IndependentVerificationError(f"E{origin} bag model artifact hash changed")
        bag = joblib.load(bag_path)["estimator"]
        coef = np.asarray(bag.coef_, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(bag.intercept_, dtype=np.float64).reshape(-1)[0])
        offset = np.asarray(score_base @ coef, dtype=np.float64).reshape(-1) + intercept
        fallback_bundle = np.load(ROOT / f"experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/E{origin}/predictions_E{origin}.npz", allow_pickle=False)
        missing_p = np.asarray(fallback_bundle["prob_missing_hgb"], dtype=np.float64)
        fallback_bundle.close()
        for candidate in CANDIDATES:
            if candidate == "UNIFORM":
                artifact = f"experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/units/POS_BAG_OFFSET_V1_E{origin}_REAL.npz"
                with np.load(ROOT / artifact, allow_pickle=False) as archive:
                    gamma = np.asarray(archive["gamma"], dtype=np.float64)
            else:
                gamma, _ = verify_gamma(origin, candidate, freeze)
            replay = missing_p.copy()
            replay[present] = expit(offset[present] + np.asarray(x[present] @ gamma, dtype=np.float64).reshape(-1))
            expected = arrays[f"E{origin}_pPOS_{candidate}"]
            if not np.array_equal(replay[~present], missing_p[~present]):
                raise IndependentVerificationError(f"E{origin}/{candidate} missing fallback is not bit exact")
            if not np.allclose(replay, expected, rtol=0.0, atol=TOL):
                raise IndependentVerificationError(f"E{origin}/{candidate} prediction replay mismatch")
            checks += 1
        expected_reference = 0.5 * arrays[f"E{origin}_pPOS_UNIFORM"] + 0.5 * p_a
        expected_reference = expected_reference.copy()
        expected_reference[missing] = p_a[missing]
        if not np.array_equal(expected_reference[missing], p_a[missing]):
            raise IndependentVerificationError(f"E{origin} primary competition fallback is not exact pA")
        assert_close(float(np.max(np.abs(expected_reference - arrays[f"E{origin}_MIX50_UNIFORM_PRIMARY"]))),
                     0.0, f"E{origin} primary reference reproduction", max_diffs)
        checks += 1
    return {"status": "PASS_INDEPENDENT_TARGET_FREE_PREDICTION_REPLAY",
            "origins_checked": len(source_origins), "probability_replays": checks,
            "maximum_numeric_difference": max(max_diffs) if max_diffs else 0.0,
            "query_targets_read": False, "test_accessed": False}


def independent_metrics(y: np.ndarray, probability: np.ndarray,
                        threshold: float = PRIMARY_THRESHOLD) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    prob = np.asarray(probability, dtype=np.float64)
    decision = prob >= threshold
    tn = int(np.count_nonzero((labels == 0) & ~decision))
    fp = int(np.count_nonzero((labels == 0) & decision))
    fn = int(np.count_nonzero((labels == 1) & ~decision))
    tp = int(np.count_nonzero((labels == 1) & decision))
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    clipped = np.minimum(np.maximum(prob, EPS), 1 - EPS)
    return {
        "n": int(labels.size), "class_0": int(np.count_nonzero(labels == 0)),
        "class_1": int(np.count_nonzero(labels == 1)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "macro_f1": float((f0 + f1) / 2),
        "positive_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "brier": float(np.mean((prob - labels) ** 2)),
        "logloss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "threshold": float(threshold), "logloss_epsilon": EPS,
    }


def independent_missing_threshold_metrics(y: np.ndarray, probability: np.ndarray,
                                          missing: np.ndarray, missing_probability: np.ndarray,
                                          threshold: float) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    prob = np.asarray(probability, dtype=np.float64)
    is_missing = np.asarray(missing, dtype=bool)
    p_missing = np.asarray(missing_probability, dtype=np.float64)
    if prob.shape != labels.shape or is_missing.shape != labels.shape or p_missing.shape != labels.shape:
        raise IndependentVerificationError("Missing-threshold metric inputs do not align")
    decision = prob >= PRIMARY_THRESHOLD
    decision[is_missing] = p_missing[is_missing] >= threshold
    tn = int(np.count_nonzero((labels == 0) & ~decision))
    fp = int(np.count_nonzero((labels == 0) & decision))
    fn = int(np.count_nonzero((labels == 1) & ~decision))
    tp = int(np.count_nonzero((labels == 1) & decision))
    base = independent_metrics(labels, prob, PRIMARY_THRESHOLD)
    base.update({
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "macro_f1": float((
            (2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0)
            + (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        ) / 2),
        "positive_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "present_threshold": PRIMARY_THRESHOLD,
        "missing_threshold": float(threshold),
    })
    base.pop("threshold")
    return base


def verify_view_metrics(name: str, y: np.ndarray, arrays: Mapping[str, np.ndarray],
                        mask: np.ndarray, origin: int, groups: Mapping[str, np.ndarray],
                        actual_view: Mapping[str, Any]) -> tuple[int, list[float]]:
    """Independently recompute all per-view metric, subgroup, and cycle payloads."""
    labels = np.asarray(y, dtype=np.int8)
    selected = np.asarray(mask, dtype=bool)
    present = np.asarray(arrays[f"E{origin}_present"], dtype=bool)
    cycles = np.asarray(arrays[f"E{origin}_cycles"], dtype=np.int64)
    p_a = np.asarray(arrays[f"E{origin}_pA"], dtype=np.float64)
    missing = ~present
    if selected.shape != labels.shape or cycles.shape != labels.shape or present.shape != labels.shape:
        raise IndependentVerificationError(f"{name} label and row arrays do not align")
    diffs: list[float] = []
    checked = 0
    counts = {
        "rows": int(selected.sum()),
        "present_rows": int((selected & present).sum()),
        "missing_rows": int((selected & ~present).sum()),
    }
    for key, expected in counts.items():
        if actual_view.get(key) != expected:
            raise IndependentVerificationError(f"{name}/{key} row-count mismatch")
        checked += 1
    component_metrics: dict[str, dict[str, Any]] = {}
    competition_metrics: dict[str, dict[str, Any]] = {}
    present_component_metrics: dict[str, dict[str, Any]] = {}
    present_competition_metrics: dict[str, dict[str, Any]] = {}
    competition_probabilities: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        p_pos = np.asarray(arrays[f"E{origin}_pPOS_{candidate}"], dtype=np.float64)
        p_comp = 0.5 * p_pos + 0.5 * p_a
        p_comp = p_comp.copy()
        p_comp[missing] = p_a[missing]
        competition_probabilities[candidate] = p_comp
        component_metrics[candidate] = independent_metrics(labels[selected], p_pos[selected])
        present_component_metrics[candidate] = independent_metrics(
            labels[selected & present], p_pos[selected & present])
        competition_metrics[candidate] = independent_metrics(labels[selected], p_comp[selected])
        present_competition_metrics[candidate] = independent_metrics(
            labels[selected & present], p_comp[selected & present])
        for field, expected in (
            ("component_full", component_metrics[candidate]),
            ("component_present_only", present_component_metrics[candidate]),
            ("competition_full", competition_metrics[candidate]),
            ("competition_present_only", present_competition_metrics[candidate]),
        ):
            _compare_metric_objects(actual_view[field][candidate], expected,
                                    f"{name}/{field}/{candidate}", diffs)
            checked += len(expected)
    full_gains = {
        candidate: competition_metrics[candidate]["macro_f1"] - competition_metrics["UNIFORM"]["macro_f1"]
        for candidate in CANDIDATES if candidate != "UNIFORM"
    }
    present_gains = {
        candidate: present_competition_metrics[candidate]["macro_f1"]
        - present_competition_metrics["UNIFORM"]["macro_f1"]
        for candidate in CANDIDATES if candidate != "UNIFORM"
    }
    for key, expected in (("competition_gain_macro_f1_vs_uniform", full_gains),
                          ("competition_present_gain_macro_f1_vs_uniform", present_gains)):
        for candidate, value in expected.items():
            assert_close(actual_view[key][candidate], value, f"{name}/{key}/{candidate}", diffs)
            checked += 1
    local_p_a = p_a[selected]
    local_missing = missing[selected]
    for candidate in CANDIDATES:
        expected = independent_missing_threshold_metrics(
            labels[selected], competition_probabilities[candidate][selected],
            local_missing, local_p_a, OPERATIONAL_MISSING_THRESHOLD)
        _compare_metric_objects(actual_view["post_selected_missing_threshold_descriptive"][candidate],
                                expected, f"{name}/operational/{candidate}", diffs)
        checked += len(expected)

    expected_subgroups: dict[str, Any] = {}
    for family in GROUPS:
        values = np.asarray(groups[family]).astype(str)
        if values.shape != labels.shape:
            raise IndependentVerificationError(f"{name}/{family} subgroup rows do not align")
        levels: dict[str, Any] = {}
        for level in sorted(set(values[selected].tolist())):
            group_mask = selected & (values == level)
            group_y = labels[group_mask]
            n0 = int(np.count_nonzero(group_y == 0))
            n1 = int(np.count_nonzero(group_y == 1))
            support = {"n": int(group_mask.sum()), "class_0": n0, "class_1": n1}
            if support["n"] < SUPPORT_N or n0 < SUPPORT_CLASS or n1 < SUPPORT_CLASS:
                levels[level] = {"status": "UNKNOWN_SUPPORT", "support": support}
                continue
            ref = independent_metrics(group_y, competition_probabilities["UNIFORM"][group_mask])
            candidates = {candidate: independent_metrics(group_y, competition_probabilities[candidate][group_mask])
                          for candidate in ("ONE_SIDED", "GLOBAL_STRENGTH")}
            levels[level] = {
                "status": "SUPPORTED", "support": support, "reference_uniform": ref,
                "candidates": candidates,
                "harm_macro_f1": {candidate: ref["macro_f1"] - candidates[candidate]["macro_f1"]
                                   for candidate in candidates},
                "harm_positive_recall": {candidate: ref["positive_recall"] - candidates[candidate]["positive_recall"]
                                          for candidate in candidates},
            }
        expected_subgroups[family] = levels
    actual_subgroups = actual_view.get("subgroups", {})
    if set(actual_subgroups) != set(expected_subgroups):
        raise IndependentVerificationError(f"{name} subgroup family set differs")
    for family, levels in expected_subgroups.items():
        actual_levels = actual_subgroups[family]
        if set(actual_levels) != set(levels):
            raise IndependentVerificationError(f"{name}/{family} subgroup levels differ")
        for level, expected in levels.items():
            actual = actual_levels[level]
            if actual.get("status") != expected["status"] or actual.get("support") != expected["support"]:
                raise IndependentVerificationError(f"{name}/{family}/{level} support/status mismatch")
            if expected["status"] == "SUPPORTED":
                _compare_metric_objects(actual["reference_uniform"], expected["reference_uniform"],
                                        f"{name}/{family}/{level}/reference", diffs)
                for candidate in expected["candidates"]:
                    _compare_metric_objects(actual["candidates"][candidate], expected["candidates"][candidate],
                                            f"{name}/{family}/{level}/{candidate}", diffs)
                    assert_close(actual["harm_macro_f1"][candidate], expected["harm_macro_f1"][candidate],
                                 f"{name}/{family}/{level}/{candidate}/harm_f1", diffs)
                    assert_close(actual["harm_positive_recall"][candidate],
                                 expected["harm_positive_recall"][candidate],
                                 f"{name}/{family}/{level}/{candidate}/harm_recall", diffs)
                    checked += 1
                checked += 1
            else:
                checked += 1

    expected_cycles: dict[str, Any] = {}
    for cycle in sorted(np.unique(cycles[selected]).tolist()):
        cycle_mask = selected & (cycles == cycle)
        cycle_y = labels[cycle_mask]
        ref = independent_metrics(cycle_y, competition_probabilities["UNIFORM"][cycle_mask])
        candidate_metrics = {
            candidate: independent_metrics(cycle_y, competition_probabilities[candidate][cycle_mask])
            for candidate in ("ONE_SIDED", "GLOBAL_STRENGTH")
        }
        expected_cycles[str(int(cycle))] = {
            "n": int(cycle_mask.sum()),
            "class_0": int(np.count_nonzero(cycle_y == 0)),
            "class_1": int(np.count_nonzero(cycle_y == 1)),
            "gain_macro_f1_vs_uniform": {
                candidate: candidate_metrics[candidate]["macro_f1"] - ref["macro_f1"]
                for candidate in candidate_metrics},
            "direction_vs_uniform": {
                candidate: ("POSITIVE" if candidate_metrics[candidate]["macro_f1"] > ref["macro_f1"]
                            else "NEGATIVE" if candidate_metrics[candidate]["macro_f1"] < ref["macro_f1"]
                            else "TIE") for candidate in candidate_metrics},
        }
    actual_cycles = actual_view.get("per_cycle", {})
    if set(actual_cycles) != set(expected_cycles):
        raise IndependentVerificationError(f"{name} cycle-view keys differ")
    for cycle, expected in expected_cycles.items():
        actual = actual_cycles[cycle]
        for key in ("n", "class_0", "class_1", "direction_vs_uniform"):
            if actual.get(key) != expected[key]:
                raise IndependentVerificationError(f"{name}/cycle-{cycle}/{key} mismatch")
        for candidate, value in expected["gain_macro_f1_vs_uniform"].items():
            assert_close(actual["gain_macro_f1_vs_uniform"][candidate], value,
                         f"{name}/cycle-{cycle}/{candidate}/gain", diffs)
            checked += 1
    return checked, diffs


def read_committed_query_labels(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict[int, np.ndarray]:
    if git("rev-parse", f"HEAD:{PRED_MANIFEST_REL}") != git("hash-object", PRED_MANIFEST_REL):
        raise IndependentVerificationError("Prediction manifest is not committed before verifier target read")
    if (manifest.get("status") != "PREDICTIONS_FROZEN_BEFORE_QUERY_TARGET_READ"
            or manifest.get("query_target_labels_read") is not False
            or manifest.get("prediction_sha256") != sha_file(ROOT / PRED_REL)):
        raise IndependentVerificationError("Prediction manifest did not freeze target-free predictions")
    identity = pd.read_csv(ROOT / TRAIN_REL, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    wanted: dict[int, str] = {}
    for origin in ORIGINS:
        rows = np.asarray(arrays[f"E{origin}_rows"], dtype=np.int64)
        ids = np.asarray(arrays[f"E{origin}_ids"]).astype(str)
        cycles = np.asarray(arrays[f"E{origin}_cycles"], dtype=np.int64)
        if (not np.array_equal(identity.iloc[rows].record_id.astype(str).to_numpy(), ids)
                or not np.array_equal(identity.iloc[rows].cycle_code.to_numpy(dtype=np.int64), cycles)):
            raise IndependentVerificationError(f"E{origin} target join identities do not match prediction IDs")
        wanted.update({int(row): str(record_id) for row, record_id in zip(rows, ids)})
    values: dict[int, int] = {}
    with (ROOT / TRAIN_REL).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_col, y_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            if row_index not in wanted:
                continue
            if record[id_col] != wanted[row_index]:
                raise IndependentVerificationError("Query target row ID differs from manifest identity")
            value = int(record[y_col])
            if value not in (0, 1):
                raise IndependentVerificationError("Query target is not binary")
            values[row_index] = value
    if values.keys() != wanted.keys():
        raise IndependentVerificationError("Some query prediction IDs have no target rows")
    return {origin: np.asarray([values[int(row)] for row in arrays[f"E{origin}_rows"]], dtype=np.int8)
            for origin in ORIGINS}


def verify_results() -> dict[str, Any]:
    prediction_audit = verify_prediction_bundle()
    manifest = json_file(PRED_MANIFEST_REL)
    with np.load(ROOT / PRED_REL, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in manifest["prediction_schema"]}
    labels = read_committed_query_labels(manifest, arrays)
    result = json_file(RESULTS_REL)
    if (result.get("campaign_id") != CAMPAIGN_ID
            or result.get("status") != "SCORED_REPEATED_DEVELOPMENT_EVALUATION"
            or result.get("predictions_manifest_sha256") != sha_file(ROOT / PRED_MANIFEST_REL)
            or result.get("prediction_sha256") != sha_file(ROOT / PRED_REL)):
        raise IndependentVerificationError("Evaluation result does not bind the committed predictions")
    views = result["views"]
    checked = 0
    diffs: list[float] = []
    for name, (origin, first, last) in VIEWS.items():
        cycles = arrays[f"E{origin}_cycles"].astype(np.int64)
        mask = (cycles >= first) & (cycles <= last)
        groups = {family: arrays[f"E{origin}_group_{family}"].astype(str) for family in GROUPS}
        added, view_diffs = verify_view_metrics(name, labels[origin], arrays, mask, origin,
                                                groups, views[name])
        checked += added
        diffs.extend(view_diffs)
    pool_pieces: list[tuple[int, np.ndarray]] = []
    for origin, members in ((67, ("E67_W4", "E67_W5")), (75, ("E75_W6",))):
        cycle_values = arrays[f"E{origin}_cycles"].astype(np.int64)
        select = np.zeros(len(cycle_values), dtype=bool)
        for member in members:
            _, first, last = VIEWS[member]
            select |= (cycle_values >= first) & (cycle_values <= last)
        pool_pieces.append((origin, select))
    pool_labels = np.concatenate([labels[origin][mask] for origin, mask in pool_pieces])
    pool_arrays: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        pool_arrays[f"E99_pPOS_{candidate}"] = np.concatenate([
            arrays[f"E{origin}_pPOS_{candidate}"][mask] for origin, mask in pool_pieces])
    for field in ("present", "pA", "cycles"):
        pool_arrays[f"E99_{field}"] = np.concatenate([
            arrays[f"E{origin}_{field}"][mask] for origin, mask in pool_pieces])
    pool_groups = {family: np.concatenate([
        arrays[f"E{origin}_group_{family}"].astype(str)[mask] for origin, mask in pool_pieces])
        for family in GROUPS}
    added, pool_diffs = verify_view_metrics(
        "UNIQUE_E67_W4_W5_PLUS_E75_W6", pool_labels, pool_arrays,
        np.ones(len(pool_labels), dtype=bool), 99, pool_groups,
        views["UNIQUE_E67_W4_W5_PLUS_E75_W6"],
    )
    checked += added
    diffs.extend(pool_diffs)
    for view_name in ("E67_H12", "E75_W6"):
        comp = views[view_name]["component_full"]
        d_uniform = comp["ONE_SIDED"]["macro_f1"] - comp["UNIFORM"]["macro_f1"]
        d_global = comp["ONE_SIDED"]["macro_f1"] - comp["GLOBAL_STRENGTH"]["macro_f1"]
        gate = result["mechanism_gate"]["checks"][view_name]
        assert_close(gate["one_sided_vs_uniform_macro_f1"], d_uniform, f"{view_name} one-sided/uniform gate", diffs)
        assert_close(gate["one_sided_vs_global_strength_macro_f1"], d_global, f"{view_name} one-sided/global gate", diffs)
        if bool(gate["pass"]) != (d_uniform > 1e-12 and d_global > 1e-12):
            raise IndependentVerificationError(f"{view_name} mechanism gate recomputation mismatch")
        checked += 2
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
                elif (entry["harm_macro_f1"]["ONE_SIDED"] > 0.01
                      or entry["harm_positive_recall"]["ONE_SIDED"] > 0.02):
                    subgroup_failures.append({
                        "group": label,
                        "harm_macro_f1": entry["harm_macro_f1"]["ONE_SIDED"],
                        "harm_positive_recall": entry["harm_positive_recall"]["ONE_SIDED"],
                    })
    expected_checks = {
        "E67_H12_gain_at_least_0.0003": gains["E67_H12"] >= 0.0003,
        "E75_W6_gain_nonnegative": gains["E75_W6"] >= 0.0,
        "unique_pool_gain_at_least_0.0003": gains["UNIQUE_E67_W4_W5_PLUS_E75_W6"] >= 0.0003,
        "each_E67_window_harm_at_most_0.003": all(value <= 0.003 for value in window_harms.values()),
        "supported_subgroups_within_guards": not subgroup_failures,
    }
    competition_gate = result["competition_gate"]
    for name, expected in expected_checks.items():
        if competition_gate["checks"].get(name) is not expected:
            raise IndependentVerificationError(f"Competition gate check mismatch: {name}")
        checked += 1
    actual_failures = competition_gate.get("supported_subgroup_failures", [])
    if (competition_gate.get("pass") is not all(expected_checks.values())
            or len(actual_failures) != len(subgroup_failures)
            or competition_gate.get("unsupported_subgroups_unknown") != subgroup_unknown):
        raise IndependentVerificationError("Competition gate summary or subgroup audit mismatch")
    for actual, expected in zip(actual_failures, subgroup_failures):
        if actual.get("group") != expected["group"]:
            raise IndependentVerificationError("Competition subgroup failure group differs")
        for key in ("harm_macro_f1", "harm_positive_recall"):
            assert_close(actual.get(key), expected[key], f"competition subgroup failure/{key}", diffs)
    for name, expected in gains.items():
        assert_close(competition_gate["gains_vs_saved_uniform_reference"][name], expected,
                     f"competition gain gate/{name}", diffs)
    for name, expected in window_harms.items():
        assert_close(competition_gate["E67_window_harms"][name], expected,
                     f"competition harm gate/{name}", diffs)
    checked += len(gains) + len(window_harms)
    expected_target_rows = len(set(np.concatenate([
        arrays[f"E{origin}_rows"] for origin in ORIGINS]).astype(int).tolist()))
    if (result.get("target_rows_read") != expected_target_rows
            or result.get("query_target_values_read_after_manifest_commit") is not True):
        raise IndependentVerificationError("Evaluation target-read accounting differs from prediction rows")
    checked += 2
    return {"status": "PASS_INDEPENDENT_RESULTS_RECOMPUTATION",
            "prediction_audit": prediction_audit,
            "metric_objects_checked": checked,
            "maximum_numeric_difference": max(diffs) if diffs else 0.0,
            "query_labels_read_after_committed_manifest": True,
            "test_accessed": False}


def _compare_metric_objects(actual: Mapping[str, Any], expected: Mapping[str, Any],
                            label: str, diffs: list[float]) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise IndependentVerificationError(f"{label} missing metric field {key}")
        value = actual[key]
        if isinstance(expected_value, list):
            if value != expected_value:
                raise IndependentVerificationError(f"{label}/{key} confusion matrix mismatch")
        else:
            assert_close(value, expected_value, f"{label}/{key}", diffs)


def assert_close(actual: Any, expected: Any, label: str, diffs: list[float]) -> None:
    if isinstance(expected, bool):
        if actual is not expected:
            raise IndependentVerificationError(f"{label}: boolean mismatch")
        return
    delta = abs(float(actual) - float(expected))
    diffs.append(delta)
    if not np.isfinite(float(actual)) or delta > TOL:
        raise IndependentVerificationError(f"{label}: {actual!r} != {expected!r} within {TOL}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--source-preflight", action="store_true")
    modes.add_argument("--source-freeze", action="store_true")
    modes.add_argument("--predictions", action="store_true")
    modes.add_argument("--results", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.source_preflight:
            result = verify_source_preflight()
        elif args.source_freeze:
            freeze = verify_source_freeze()
            result = {"status": "PASS_INDEPENDENT_SOURCE_FREEZE", "freeze_signature": freeze["freeze_signature"],
                      "official_fits_authorized": False}
        elif args.predictions:
            result = verify_prediction_bundle()
        else:
            result = verify_results()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        return 0
    except (IndependentVerificationError, OLD_EVAL.EvaluationError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
