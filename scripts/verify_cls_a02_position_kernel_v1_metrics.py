from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.a02_position_kernel_v1 import load_npz_allowlisted


OFFICIAL_ROOT = Path(os.environ.get("KASDAD_TP2_SOURCE_ROOT", REPO_ROOT.parent / "TP 2")).resolve()
REPORT_ROOT = REPO_ROOT / "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1"
RUN_ROOT = REPO_ROOT / "experiments/runs/simplification_v1/CLS_POST_BAG2_NEXT_V1"
RESULTS_PATH = REPORT_ROOT / "RESULTS.json"
VERIFY_PATH = REPORT_ROOT / "METRIC_VERIFICATION.json"
PRELABEL_PATH = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
ATTEMPT_PATH = REPORT_ROOT / "SCORING_ATTEMPT.json"
FREEZE_PATH = REPORT_ROOT / "FREEZE.json"
TRAIN_PATH = OFFICIAL_ROOT / "data/raw/classification/Train.csv"
SUPPORT_N = 200
SUPPORT_CLASS = 30
F1_HARM_MAX = 0.02
RECALL_HARM_MAX = 0.03

OUTPUT_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "missing_mask", "entry_friction",
    "text_load_group", "p_sparse25", "p_a02_original", "p_pos_original",
    "p_a02_candidate", "champion_score", "candidate_score", "champion_class",
    "candidate_class", "m0", "neighbor_count", "neighbor_source_rows", "neighbor_ids",
    "neighbor_kernel_similarity", "neighbor_weights", "position_kernel_positive_mass",
    "correction_raw", "correction", "mass_error", "neighbor_overlap_fraction",
)


class MetricVerificationError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jread(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _committed_and_clean(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                             text=True, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT,
                           text=True, capture_output=True).returncode == 0
    return tracked and clean


def _metrics_independent(y: np.ndarray, p: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int8)
    if len(y) == 0 or p.shape != y.shape or pred.shape != y.shape:
        raise MetricVerificationError("Metric input rows are misaligned")
    cm = confusion_matrix(y, pred, labels=(0, 1))
    if len(np.unique(y)) != 2:
        auc = ap = None
    else:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    p_safe = np.minimum(1.0 - 1e-15, np.maximum(1e-15, p))
    return {
        "n": int(y.size),
        "positive_count": int(y.sum()),
        "negative_count": int(y.size - int(y.sum())),
        "macro_f1": float(f1_score(y, pred, labels=(0, 1), average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": auc,
        "average_precision": ap,
        "log_loss_eps_1e_minus_15": float(log_loss(y, p_safe, labels=(0, 1))),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix_labels_0_1": cm.astype(int).tolist(),
    }


def _assert_metric_dict(actual: dict[str, Any], expected: dict[str, Any], where: str) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if actual_value is None or not np.isfinite(actual_value) or not np.isclose(actual_value, expected_value, rtol=0, atol=1e-12):
                raise MetricVerificationError(f"Metric mismatch {where}.{key}: {actual_value} != {expected_value}")
        elif actual_value != expected_value:
            raise MetricVerificationError(f"Metric mismatch {where}.{key}: {actual_value} != {expected_value}")


def _bundle(origin: int, arm: str, prelabel: dict[str, Any]) -> dict[str, np.ndarray]:
    unit_id = f"E{origin}_{arm}"
    path = RUN_ROOT / f"{unit_id}/predictions.npz"
    receipt_path = RUN_ROOT / f"{unit_id}/receipt.json"
    verified = prelabel.get("four_unit_receipts_and_outputs", {}).get(unit_id, {})
    if (not isinstance(verified.get("artifact_sha256"), str)
            or sha_file(path) != verified.get("artifact_sha256")):
        raise MetricVerificationError(f"Metric verifier input is not the independently verified artifact: {unit_id}")
    if (not isinstance(verified.get("receipt_sha256"), str)
            or sha_file(receipt_path) != verified.get("receipt_sha256")):
        raise MetricVerificationError(f"Metric verifier receipt differs from prelabel verification: {unit_id}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(OUTPUT_MEMBERS):
            raise MetricVerificationError(f"Prediction output contains target/unexpected members: {unit_id}")
    return load_npz_allowlisted(path, allowed_members=OUTPUT_MEMBERS)


def _target(bundle: dict[str, np.ndarray], frame: pd.DataFrame) -> np.ndarray:
    raw = frame.iloc[bundle["score_rows"].astype(np.int64)]
    ids = raw["record_id"].astype("string").fillna("").astype(str).to_numpy()
    if not np.array_equal(ids.astype(bundle["score_record_ids"].dtype), bundle["score_record_ids"]):
        raise MetricVerificationError("Independent target row IDs do not match candidate bundle")
    if not np.array_equal(raw["cycle_code"].to_numpy(dtype=np.int64), bundle["score_cycles"].astype(np.int64)):
        raise MetricVerificationError("Independent target cycles do not match candidate bundle")
    y_numeric = pd.to_numeric(raw["artifact_signal"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y_numeric).all() or not np.isin(y_numeric, [0.0, 1.0]).all():
        raise MetricVerificationError("Official Classification Train target is not integer 0/1")
    return y_numeric.astype(np.int8)


def _combine_unique(first: dict[str, np.ndarray], second: dict[str, np.ndarray], y_first: np.ndarray,
                    y_second: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    early = (first["score_cycles"] >= 68) & (first["score_cycles"] <= 75)
    late = (second["score_cycles"] >= 76) & (second["score_cycles"] <= 79)
    ids = np.concatenate((first["score_record_ids"][early].astype(str), second["score_record_ids"][late].astype(str)))
    if len(ids) != 20903 or len(set(ids.tolist())) != 20903:
        raise MetricVerificationError("Independent unique pool is not 20,903 distinct rows")
    result = {}
    for name in ("score_rows", "score_record_ids", "score_cycles", "missing_mask", "entry_friction", "text_load_group",
                 "p_sparse25", "p_a02_original", "p_pos_original", "p_a02_candidate", "champion_score",
                 "candidate_score", "champion_class", "candidate_class", "m0", "neighbor_count", "correction_raw",
                 "correction", "mass_error", "neighbor_overlap_fraction"):
        result[name] = np.concatenate((first[name][early], second[name][late]))
    y = np.concatenate((y_first[early], y_second[late]))
    return result, y


def _score_view(real: dict[str, np.ndarray], shuffled: dict[str, np.ndarray],
                y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    champ = _metrics_independent(y[mask], real["champion_score"][mask], real["champion_class"][mask])
    real_metrics = _metrics_independent(y[mask], real["candidate_score"][mask], real["candidate_class"][mask])
    shuffled_metrics = _metrics_independent(y[mask], shuffled["candidate_score"][mask], shuffled["candidate_class"][mask])
    return {
        "full_champion": champ,
        "real": real_metrics,
        "shuffled": shuffled_metrics,
        "gain_real_vs_champion": real_metrics["macro_f1"] - champ["macro_f1"],
        "gain_shuffled_vs_champion": shuffled_metrics["macro_f1"] - champ["macro_f1"],
        "gain_real_vs_shuffled": real_metrics["macro_f1"] - shuffled_metrics["macro_f1"],
    }


def _subgroup_check(arr: dict[str, np.ndarray], y: np.ndarray, view_mask: np.ndarray,
                    expected: dict[str, Any]) -> tuple[int, int, list[bool]]:
    definitions: dict[str, list[tuple[str, np.ndarray]]] = {
        "availability": [("missing", arr["missing_mask"]), ("present", ~arr["missing_mask"])],
        "text_load_quartile": [(str(int(v)), arr["text_load_group"] == v) for v in sorted(set(arr["text_load_group"].tolist()))],
        "entry_friction": [(str(int(v)), arr["entry_friction"] == v) for v in sorted(set(arr["entry_friction"].tolist()))],
    }
    support_count = unknown_count = 0
    guards: list[bool] = []
    for family, groups in definitions.items():
        for name, membership in groups:
            mask = view_mask & membership
            gy = y[mask]
            n = int(len(gy))
            pos, neg = int(gy.sum()), n - int(gy.sum())
            label = "SUPPORTED" if n >= SUPPORT_N and pos >= SUPPORT_CLASS and neg >= SUPPORT_CLASS else "UNKNOWN"
            recorded = expected[family][name]
            if recorded.get("n") != n or recorded.get("positive_count") != pos or recorded.get("negative_count") != neg:
                raise MetricVerificationError(f"Subgroup support counts differ at {family}/{name}")
            if recorded.get("support") != label:
                raise MetricVerificationError(f"Subgroup support status differs at {family}/{name}")
            if label == "SUPPORTED":
                support_count += 1
                champ = _metrics_independent(gy, arr["champion_score"][mask], arr["champion_class"][mask])
                real = _metrics_independent(gy, arr["candidate_score"][mask], arr["candidate_class"][mask])
                gain = real["macro_f1"] - champ["macro_f1"]
                recall_harm = champ["positive_recall"] - real["positive_recall"]
                passed = gain >= -F1_HARM_MAX and recall_harm <= RECALL_HARM_MAX
                _assert_metric_dict(recorded["champion"], champ, f"subgroup.{family}.{name}.champion")
                _assert_metric_dict(recorded["real"], real, f"subgroup.{family}.{name}.real")
                if not np.isclose(recorded.get("gain_real_vs_champion_macro_f1"), gain, rtol=0, atol=1e-12):
                    raise MetricVerificationError(f"Subgroup F1 gain differs at {family}/{name}")
                if not np.isclose(recorded.get("positive_recall_harm_vs_champion"), recall_harm, rtol=0, atol=1e-12):
                    raise MetricVerificationError(f"Subgroup recall harm differs at {family}/{name}")
                if recorded.get("guard_pass") is not passed:
                    raise MetricVerificationError(f"Subgroup guard differs at {family}/{name}")
                guards.append(passed)
            else:
                unknown_count += 1
    return support_count, unknown_count, guards


def _cycle_signs(arr: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    result: dict[str, Any] = {}
    for cycle in sorted(np.unique(arr["score_cycles"][mask]).tolist()):
        one = mask & (arr["score_cycles"] == cycle)
        champ = _metrics_independent(y[one], arr["champion_score"][one], arr["champion_class"][one])["macro_f1"]
        real = _metrics_independent(y[one], arr["candidate_score"][one], arr["candidate_class"][one])["macro_f1"]
        difference = real - champ
        result[str(int(cycle))] = {
            "n": int(one.sum()),
            "f1_real_minus_champion": difference,
            "direction": "POSITIVE" if difference > 1e-12 else "NEGATIVE" if difference < -1e-12 else "TIED",
        }
    directions = [value["direction"] for value in result.values()]
    result["direction_counts"] = {key: directions.count(key) for key in ("POSITIVE", "NEGATIVE", "TIED")}
    return result


def _assert_per_unit_diagnostics(actual: dict[str, Any], arr: dict[str, np.ndarray], unit_id: str) -> None:
    supported = (~arr["missing_mask"]) & (arr["m0"] > 0.0)
    neighbors = arr["neighbor_count"] > 0
    denominator = max(1, int(supported.sum()))
    expected = {
        "rows": int(len(arr["score_rows"])),
        "missing_rows": int(arr["missing_mask"].sum()),
        "present_supported_rows": int(supported.sum()),
        "mean_neighbor_overlap_supported": float(np.mean(arr["neighbor_overlap_fraction"][neighbors])) if np.any(neighbors) else 0.0,
        "correction_clip_rows": int(np.sum(np.abs(arr["correction_raw"]) > 0.05)),
        "correction_clip_rate_supported_present": float(np.sum((np.abs(arr["correction_raw"]) > 0.05) & supported) / denominator),
        "max_abs_mass_error": float(np.max(np.abs(arr["mass_error"]))),
    }
    for key, value in expected.items():
        observed = actual.get(key)
        if isinstance(value, float):
            if observed is None or not np.isclose(observed, value, rtol=0, atol=1e-12):
                raise MetricVerificationError(f"Per-unit diagnostic mismatch {unit_id}.{key}")
        elif observed != value:
            raise MetricVerificationError(f"Per-unit diagnostic mismatch {unit_id}.{key}")


def verify_metrics() -> dict[str, Any]:
    if VERIFY_PATH.exists():
        raise MetricVerificationError("Metric verifier output already exists; do not overwrite")
    results = jread(RESULTS_PATH)
    prelabel = jread(PRELABEL_PATH)
    attempt = jread(ATTEMPT_PATH)
    freeze = jread(FREEZE_PATH)
    if results.get("status") != "SCORED_EXPLORATORY_WITH_PROTOCOL_DEVIATION":
        raise MetricVerificationError("Result status changed or is missing")
    if results.get("original_prelabel_boundary_compliant") is not False or results.get("scientific_qualification") is not False:
        raise MetricVerificationError("Protocol deviation or scientific qualification flags were changed")
    if prelabel.get("status") != "PASS_AMENDED_PRELABEL_VERIFICATION" or not prelabel.get("model_construction_training_boundary_verified"):
        raise MetricVerificationError("Amended prelabel verification does not pass")
    if not _committed_and_clean(FREEZE_PATH) or not _committed_and_clean(PRELABEL_PATH):
        raise MetricVerificationError("Freeze and prelabel verification must remain committed and unchanged")
    source_commit = freeze.get("source_commit")
    if (prelabel.get("source_commit") != source_commit
            or results.get("source_commit") != source_commit
            or attempt.get("source_commit") != source_commit
            or prelabel.get("official_source_commit") != freeze.get("official_source_commit")
            or results.get("official_source_commit") != freeze.get("official_source_commit")):
        raise MetricVerificationError("Source commit identities differ across frozen result lineage")
    if attempt.get("status") != "AUTHORIZED_SCORE_STARTED" or attempt.get("prelabel_verification_sha256") != sha_file(PRELABEL_PATH):
        raise MetricVerificationError("Scoring attempt is not bound to committed prelabel evidence")
    if results.get("prelabel_verification_sha256") != sha_file(PRELABEL_PATH) or results.get("freeze_sha256") != sha_file(FREEZE_PATH):
        raise MetricVerificationError("Scored result lineage differs from the frozen prelabel boundary")

    expected_train_hash = freeze.get("input_hashes", {}).get("OFFICIAL::data/raw/classification/Train.csv")
    if (not isinstance(expected_train_hash, str) or sha_file(TRAIN_PATH) != expected_train_hash
            or attempt.get("official_train_sha256") != expected_train_hash
            or results.get("official_train_sha256") != expected_train_hash):
        raise MetricVerificationError("Official Train differs from its frozen prelabel/scoring input hash")
    expected_bundles = {
        unit_id: {
            "prediction_sha256": item.get("artifact_sha256"),
            "receipt_sha256": item.get("receipt_sha256"),
        }
        for unit_id, item in prelabel.get("four_unit_receipts_and_outputs", {}).items()
    }
    if (set(expected_bundles) != {"E67_REAL", "E67_SHUFFLED", "E75_REAL", "E75_SHUFFLED"}
            or attempt.get("candidate_bundles") != expected_bundles
            or results.get("candidate_bundles") != expected_bundles):
        raise MetricVerificationError("Scoring inputs do not match all four prelabel-verified artifacts")

    frame = pd.read_csv(TRAIN_PATH, usecols=["record_id", "cycle_code", "artifact_signal"], dtype={"record_id": "string"})
    if sha_file(TRAIN_PATH) != expected_train_hash:
        raise MetricVerificationError("Official Train changed while independent labels were read")
    e67r, e67s = _bundle(67, "REAL", prelabel), _bundle(67, "SHUFFLED", prelabel)
    e75r, e75s = _bundle(75, "REAL", prelabel), _bundle(75, "SHUFFLED", prelabel)
    y67r, y67s, y75r, y75s = _target(e67r, frame), _target(e67s, frame), _target(e75r, frame), _target(e75s, frame)
    if not np.array_equal(y67r, y67s) or not np.array_equal(y75r, y75s):
        raise MetricVerificationError("REAL/SHUFFLED labels differ on a row binding")
    unique_r, uy = _combine_unique(e67r, e75r, y67r, y75r)
    unique_s, uys = _combine_unique(e67s, e75s, y67s, y75s)
    if not np.array_equal(uy, uys):
        raise MetricVerificationError("Unique-pool REAL/SHUFFLED labels differ")

    view_specs = {
        "UNIQUE_PRIMARY": (unique_r, unique_s, uy, np.ones(len(uy), dtype=bool)),
        "E67_H12": (e67r, e67s, y67r, (e67r["score_cycles"] >= 68) & (e67r["score_cycles"] <= 79)),
        "E67_W4": (e67r, e67s, y67r, (e67r["score_cycles"] >= 68) & (e67r["score_cycles"] <= 71)),
        "E67_W5": (e67r, e67s, y67r, (e67r["score_cycles"] >= 72) & (e67r["score_cycles"] <= 75)),
        "E67_W6": (e67r, e67s, y67r, (e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79)),
        "E75_W6": (e75r, e75s, y75r, (e75r["score_cycles"] >= 76) & (e75r["score_cycles"] <= 79)),
    }
    checked_views = {}
    subgroup_counts = {"supported": 0, "unknown": 0, "failed": 0}
    all_group_guards: list[bool] = []
    for name, (real_arr, shuffled_arr, y, mask) in view_specs.items():
        m = _score_view(real_arr, shuffled_arr, y, mask)
        recorded = results["views"][name]
        if recorded.get("n") != int(mask.sum()):
            raise MetricVerificationError(f"View row count differs for {name}")
        _assert_metric_dict(recorded["full_champion"], m["full_champion"], f"{name}.champion")
        _assert_metric_dict(recorded["real"], m["real"], f"{name}.real")
        _assert_metric_dict(recorded["shuffled"], m["shuffled"], f"{name}.shuffled")
        if not np.isclose(recorded.get("gain_real_vs_champion_macro_f1"), m["gain_real_vs_champion"], rtol=0, atol=1e-12):
            raise MetricVerificationError(f"Macro F1 gain differs for {name}")
        if not np.isclose(recorded.get("gain_shuffled_vs_champion_macro_f1"), m["gain_shuffled_vs_champion"], rtol=0, atol=1e-12):
            raise MetricVerificationError(f"Shuffled/champion Macro F1 gain differs for {name}")
        if not np.isclose(recorded.get("gain_real_vs_shuffled_macro_f1"), m["gain_real_vs_shuffled"], rtol=0, atol=1e-12):
            raise MetricVerificationError(f"Real/shuffled Macro F1 difference differs for {name}")
        expected_row_hash = hashlib.sha256("\n".join(real_arr["score_record_ids"][mask].astype(str)).encode("utf-8")).hexdigest()
        if recorded.get("row_id_sha256") != expected_row_hash:
            raise MetricVerificationError(f"View row identity hash differs for {name}")
        s, u, guards = _subgroup_check(real_arr, y, mask, results["subgroups"][name])
        subgroup_counts["supported"] += s
        subgroup_counts["unknown"] += u
        subgroup_counts["failed"] += sum(not value for value in guards)
        all_group_guards.extend(guards)
        checked_views[name] = {"n": m["full_champion"]["n"],
                               "real_macro_f1_gain": m["gain_real_vs_champion"],
                               "real_minus_shuffled_macro_f1": m["gain_real_vs_shuffled"]}

    if not all_group_guards:
        raise MetricVerificationError("No supported subgroup guard was available")
    if results.get("subgroup_guard_summary") != {
        "supported_cells": subgroup_counts["supported"],
        "failed_cells": subgroup_counts["failed"],
        "unknown_cells": subgroup_counts["unknown"],
        "pass": subgroup_counts["failed"] == 0,
    }:
        raise MetricVerificationError("Subgroup guard summary differs from independent reconstruction")
    supported_guard_pass = all(all_group_guards)
    unique_gain = checked_views["UNIQUE_PRIMARY"]["real_macro_f1_gain"]
    e67_gain = checked_views["E67_H12"]["real_macro_f1_gain"]
    e75_gain = checked_views["E75_W6"]["real_macro_f1_gain"]
    real_vs_shuffled_pool = checked_views["UNIQUE_PRIMARY"]["real_minus_shuffled_macro_f1"]
    real_vs_shuffled_e75 = checked_views["E75_W6"]["real_minus_shuffled_macro_f1"]
    gates = {
        "real_unique_pool_gain_ge_0_0005": unique_gain >= 0.0005,
        "real_e67_h12_gain_nonnegative": e67_gain >= 0.0,
        "real_e75_w6_gain_nonnegative": e75_gain >= 0.0,
        "real_vs_shuffled_unique_gain_ge_0_0002": real_vs_shuffled_pool >= 0.0002,
        "real_vs_shuffled_e75_w6_gain_nonnegative": real_vs_shuffled_e75 >= 0.0,
        "supported_legacy_guards_pass": supported_guard_pass,
    }
    if gates != results["scientific_performance_gates"]:
        raise MetricVerificationError("Frozen scientific performance gate booleans do not match independent calculation")
    pool_policy = unique_gain > 0.0 or (e75_gain >= 0.0005 and unique_gain >= -0.0005)
    missing_invariant = all(np.array_equal(b["p_a02_candidate"][b["missing_mask"]], b["p_a02_original"][b["missing_mask"]])
                            for b in (e67r, e67s, e75r, e75s))
    eligibility = pool_policy and supported_guard_pass and missing_invariant
    expected_eligibility = "ELIGIBLE_EXPLORATORY_ONLY" if eligibility else "NOT_ELIGIBLE_EXPLORATORY"
    if results["exploratory_eligibility"].get("status") != expected_eligibility:
        raise MetricVerificationError("Exploratory eligibility status does not match independent calculation")
    eligibility_flags = {
        "pool_policy_pass": bool(pool_policy),
        "supported_group_harm_policy_pass": supported_guard_pass,
        "missing_invariant_pass": bool(missing_invariant),
        "independent_prelabel_qa_pass": True,
        "training_boundary_pass": True,
    }
    for key, value in eligibility_flags.items():
        if results["exploratory_eligibility"].get(key) is not value:
            raise MetricVerificationError(f"Exploratory eligibility field {key} differs from independent calculation")
    if results["exploratory_eligibility"].get("scientific_qualification") is not False:
        raise MetricVerificationError("Exploratory eligibility rewrote scientific qualification")

    e67_w6_rows = int(np.sum((e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79)))
    if results.get("overlap") != {
        "unique_primary_rows": 20903,
        "e67_e75_w6_rows": e67_w6_rows,
        "e67_e75_w6_counted_once_in_unique_pool": True,
    }:
        raise MetricVerificationError("Overlap and unique denominator metadata differ from independent row binding")
    expected_cycles = {
        "E67_REAL_MINUS_CHAMPION": _cycle_signs(e67r, y67r),
        "E67_SHUFFLED_MINUS_CHAMPION": _cycle_signs(e67s, y67s),
        "E75_REAL_MINUS_CHAMPION": _cycle_signs(e75r, y75r),
        "E75_SHUFFLED_MINUS_CHAMPION": _cycle_signs(e75s, y75s),
        "UNIQUE_REAL_MINUS_CHAMPION": _cycle_signs(unique_r, uy),
    }
    if results.get("cycle_signs") != expected_cycles:
        raise MetricVerificationError("Per-cycle direction/count result differs from independent reconstruction")
    for origin, real, shuffled in ((67, e67r, e67s), (75, e75r, e75s)):
        for arm, bundle in (("REAL", real), ("SHUFFLED", shuffled)):
            unit_id = f"E{origin}_{arm}"
            _assert_per_unit_diagnostics(results.get("per_unit_diagnostics", {}).get(unit_id, {}), bundle, unit_id)

    report = {
        "status": "PASS_INDEPENDENT_METRIC_AND_GATE_RECONSTRUCTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_sha256": sha_file(RESULTS_PATH),
        "prelabel_sha256": sha_file(PRELABEL_PATH),
        "scoring_attempt_sha256": sha_file(ATTEMPT_PATH),
        "official_train_sha256": sha_file(TRAIN_PATH),
        "candidate_bundles": attempt.get("candidate_bundles"),
        "scored_views_checked": checked_views,
        "subgroup_support_counts": subgroup_counts,
        "scientific_performance_gates_recomputed": gates,
        "exploratory_policy_recomputed": {"pool_policy_pass": bool(pool_policy),
                                          "supported_groups_pass": supported_guard_pass,
                                          "missing_invariant_pass": bool(missing_invariant),
                                          "eligibility": expected_eligibility,
                                          "scientific_qualification": False},
        "query_labels_accessed_after_boundary_for_metric_verification": True,
        "target_member_read_from_npz": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_accessed": False,
        "original_prelabel_boundary_compliant": False,
        "scientific_qualification": False,
    }
    with VERIFY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return report


if __name__ == "__main__":
    try:
        print(json.dumps(verify_metrics(), sort_keys=True, ensure_ascii=False))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
