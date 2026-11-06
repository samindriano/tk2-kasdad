from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.a02_position_kernel_v1 import classification_metrics, load_npz_allowlisted


OFFICIAL_ROOT = Path(os.environ.get("KASDAD_TP2_SOURCE_ROOT", REPO_ROOT.parent / "TP 2")).resolve()
REPORT_ROOT = REPO_ROOT / "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1"
RUN_ROOT = REPO_ROOT / "experiments/runs/simplification_v1/CLS_POST_BAG2_NEXT_V1"
FREEZE = REPORT_ROOT / "FREEZE.json"
PRELABEL = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
SCORE_ATTEMPT = REPORT_ROOT / "SCORING_ATTEMPT.json"
RESULTS = REPORT_ROOT / "RESULTS.json"
RESULTS_MD = REPORT_ROOT / "RESULTS.md"
METRIC_VERIFICATION = REPORT_ROOT / "METRIC_VERIFICATION.json"
TRAIN = OFFICIAL_ROOT / "data/raw/classification/Train.csv"
MISSING_THRESHOLD = 0.53230976129477
SUPPORT_N = 200
SUPPORT_CLASS = 30
GROUP_F1_HARM_MAX = 0.02
GROUP_RECALL_HARM_MAX = 0.03

A02_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
    "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
    "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction",
    "correction_unclipped",
)
POS_MEMBERS = ("score_rows", "score_record_ids", "score_cycles", "present_mask", "prob_position")
OUTPUT_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "missing_mask", "entry_friction",
    "text_load_group", "p_sparse25", "p_a02_original", "p_pos_original",
    "p_a02_candidate", "champion_score", "candidate_score", "champion_class",
    "candidate_class", "m0", "neighbor_count", "neighbor_source_rows", "neighbor_ids",
    "neighbor_kernel_similarity", "neighbor_weights", "position_kernel_positive_mass",
    "correction_raw", "correction", "mass_error", "neighbor_overlap_fraction",
)


class ScoreError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_unit(origin: int, arm: str) -> dict[str, np.ndarray]:
    return load_npz_allowlisted(RUN_ROOT / f"E{origin}_{arm}/predictions.npz",
                                allowed_members=OUTPUT_MEMBERS)


def _targets_for(bundle: dict[str, np.ndarray], train: pd.DataFrame) -> np.ndarray:
    rows = bundle["score_rows"].astype(np.int64)
    raw = train.iloc[rows]
    ids = raw["record_id"].astype("string").fillna("").astype(str).to_numpy()
    cycles = raw["cycle_code"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids.astype(bundle["score_record_ids"].dtype), bundle["score_record_ids"]):
        raise ScoreError("Score row record IDs differ from the official Train target rows")
    if not np.array_equal(cycles, bundle["score_cycles"].astype(np.int64)):
        raise ScoreError("Score row cycle codes differ from the official Train target rows")
    y_numeric = pd.to_numeric(raw["artifact_signal"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y_numeric).all() or not np.isin(y_numeric, [0.0, 1.0]).all():
        raise ScoreError("Official Classification Train target is not integer 0/1")
    return y_numeric.astype(np.int8)


def _assert_prelabel_artifact_hash(prelabel: dict[str, Any], unit_id: str, path: Path) -> None:
    """Bind the current bundle to the exact artifact independently verified prelabel."""
    verified = prelabel.get("four_unit_receipts_and_outputs", {}).get(unit_id, {})
    expected = verified.get("artifact_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or sha_file(path) != expected:
        raise ScoreError(f"Prediction bundle is not the independently verified prelabel artifact: {unit_id}")


def _view_from_arrays(
    name: str,
    real: dict[str, np.ndarray],
    shuffled: dict[str, np.ndarray],
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if not np.array_equal(real["score_record_ids"][mask], shuffled["score_record_ids"][mask]):
        raise ScoreError(f"Real/shuffled row identity differs in {name}")
    if not np.array_equal(real["champion_score"][mask], shuffled["champion_score"][mask]):
        raise ScoreError(f"Full champion comparator differs by arm in {name}")
    if not np.array_equal(real["champion_class"][mask], shuffled["champion_class"][mask]):
        raise ScoreError(f"Full champion class comparator differs by arm in {name}")
    y = labels[mask]
    champ = classification_metrics(y, real["champion_score"][mask], real["champion_class"][mask])
    real_metric = classification_metrics(y, real["candidate_score"][mask], real["candidate_class"][mask])
    shuffled_metric = classification_metrics(y, shuffled["candidate_score"][mask], shuffled["candidate_class"][mask])
    return {
        "n": int(mask.sum()),
        "row_id_sha256": hashlib.sha256("\n".join(real["score_record_ids"][mask].astype(str)).encode("utf-8")).hexdigest(),
        "full_champion": champ,
        "real": real_metric,
        "shuffled": shuffled_metric,
        "gain_real_vs_champion_macro_f1": real_metric["macro_f1"] - champ["macro_f1"],
        "gain_shuffled_vs_champion_macro_f1": shuffled_metric["macro_f1"] - champ["macro_f1"],
        "gain_real_vs_shuffled_macro_f1": real_metric["macro_f1"] - shuffled_metric["macro_f1"],
    }


def _concat_unique(e67: dict[str, np.ndarray], e75: dict[str, np.ndarray], y67: np.ndarray,
                   y75: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    early = (e67["score_cycles"].astype(np.int64) >= 68) & (e67["score_cycles"].astype(np.int64) <= 75)
    late = (e75["score_cycles"].astype(np.int64) >= 76) & (e75["score_cycles"].astype(np.int64) <= 79)
    ids = np.concatenate([e67["score_record_ids"][early].astype(str), e75["score_record_ids"][late].astype(str)])
    if len(ids) != 20903 or len(set(ids.tolist())) != 20903:
        raise ScoreError(f"Unique primary pool must have 20,903 distinct IDs, got {len(set(ids.tolist()))}")
    keys = ("score_rows", "score_record_ids", "score_cycles", "missing_mask", "entry_friction",
            "text_load_group", "p_sparse25", "p_a02_original", "p_pos_original", "p_a02_candidate",
            "champion_score", "candidate_score", "champion_class", "candidate_class", "m0",
            "neighbor_count", "correction_raw", "correction", "mass_error", "neighbor_overlap_fraction")
    merged = {key: np.concatenate([e67[key][early], e75[key][late]]) for key in keys}
    y = np.concatenate([y67[early], y75[late]])
    if not np.array_equal(merged["score_record_ids"].astype(str), ids):
        raise ScoreError("Unique pool ID assembly changed order")
    return merged, y


def _subgroups(arrays: dict[str, np.ndarray], labels: np.ndarray, view_name: str,
               view_mask: np.ndarray | None = None) -> dict[str, Any]:
    if view_mask is None:
        view_mask = np.ones(len(labels), dtype=bool)
    result: dict[str, Any] = {}
    group_defs: dict[str, list[tuple[str, np.ndarray]]] = {
        "availability": [("missing", arrays["missing_mask"]), ("present", ~arrays["missing_mask"])],
    }
    for key, label in (("text_load_group", "text_load_quartile"), ("entry_friction", "entry_friction")):
        values = arrays[key]
        group_defs[label] = [(str(int(value)), values == value) for value in sorted(set(values.tolist()))]
    for family, groups in group_defs.items():
        result[family] = {}
        for group_name, group_mask in groups:
            mask = view_mask & group_mask
            y = labels[mask]
            n = len(y)
            pos = int(y.sum())
            neg = n - pos
            item = {"n": int(n), "positive_count": pos, "negative_count": neg,
                    "support": "SUPPORTED" if n >= SUPPORT_N and pos >= SUPPORT_CLASS and neg >= SUPPORT_CLASS else "UNKNOWN"}
            if item["support"] == "SUPPORTED":
                champ = classification_metrics(y, arrays["champion_score"][mask], arrays["champion_class"][mask])
                real = classification_metrics(y, arrays["candidate_score"][mask], arrays["candidate_class"][mask])
                item.update({
                    "champion": champ,
                    "real": real,
                    "gain_real_vs_champion_macro_f1": real["macro_f1"] - champ["macro_f1"],
                    "positive_recall_harm_vs_champion": champ["positive_recall"] - real["positive_recall"],
                    "guard_pass": (real["macro_f1"] - champ["macro_f1"] >= -GROUP_F1_HARM_MAX
                                   and champ["positive_recall"] - real["positive_recall"] <= GROUP_RECALL_HARM_MAX),
                })
            result[family][group_name] = item
    return result


def _cycle_signs(arrays: dict[str, np.ndarray], labels: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(labels), dtype=bool)
    result: dict[str, Any] = {}
    for cycle in sorted(np.unique(arrays["score_cycles"][mask]).tolist()):
        one = mask & (arrays["score_cycles"] == cycle)
        y = labels[one]
        champ = classification_metrics(y, arrays["champion_score"][one], arrays["champion_class"][one])["macro_f1"]
        real = classification_metrics(y, arrays["candidate_score"][one], arrays["candidate_class"][one])["macro_f1"]
        difference = real - champ
        result[str(int(cycle))] = {"n": int(one.sum()), "f1_real_minus_champion": difference,
                                   "direction": "POSITIVE" if difference > 1e-12 else "NEGATIVE" if difference < -1e-12 else "TIED"}
    directions = [x["direction"] for x in result.values()]
    result["direction_counts"] = {key: directions.count(key) for key in ("POSITIVE", "NEGATIVE", "TIED")}
    return result


def _md(result: dict[str, Any]) -> str:
    views = result["views"]
    lines = [
        "# CLS A02 position-kernel pilot results",
        "",
        "Exploratory result only. The original prelabel boundary was violated by the disclosed E67 cache-member access; this run is not outcome-blind and is not scientifically qualified.",
        "",
        "| View | N | Champion Macro F1 | REAL Macro F1 | REAL gain | SHUFFLED Macro F1 | REAL − SHUFFLED |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("UNIQUE_PRIMARY", "E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6"):
        item = views[name]
        lines.append(f"| {name} | {item['n']} | {item['full_champion']['macro_f1']:.6f} | {item['real']['macro_f1']:.6f} | {item['gain_real_vs_champion_macro_f1']:+.6f} | {item['shuffled']['macro_f1']:.6f} | {item['gain_real_vs_shuffled_macro_f1']:+.6f} |")
    lines.extend(["", "## Decision gates", "",
                  f"- Scientific performance gates: **{'PASS' if all(result['scientific_performance_gates'].values()) else 'FAIL'}** (qualification remains **FAIL** because of the recorded protocol deviation).",
                  f"- Exploratory eligibility: **{result['exploratory_eligibility']['status']}**.",
                  f"- Supported subgroup cells: {result['subgroup_guard_summary']['supported_cells']}; guard failures: {result['subgroup_guard_summary']['failed_cells']}; unsupported cells remain UNKNOWN.",
                  f"- E67/E75 overlapping W6 rows: {result['overlap']['e67_e75_w6_rows']}; unique primary pool rows: {result['overlap']['unique_primary_rows']}.",
                  "- The operational missing-stream threshold was previously learned from global OOF, so these paired temporal views are adaptive development evidence, not a fresh holdout.",
                  "- No Kaggle/Test/E79 action was taken.", ""])
    return "\n".join(lines)


def score_once() -> dict[str, Any]:
    if SCORE_ATTEMPT.exists() or RESULTS.exists() or RESULTS_MD.exists():
        raise ScoreError("A score attempt or result already exists; do not score again")
    freeze = load_json(FREEZE)
    prelabel = load_json(PRELABEL)
    if prelabel.get("status") != "PASS_AMENDED_PRELABEL_VERIFICATION":
        raise ScoreError("Independent prelabel verification is not PASS")
    if prelabel.get("original_prelabel_boundary_compliant") is not False or prelabel.get("scientific_qualification") is not False:
        raise ScoreError("Protocol-deviation flags were changed")
    if not prelabel.get("model_construction_training_boundary_verified"):
        raise ScoreError("Model-construction training boundary is not independently verified")
    if prelabel.get("freeze_sha256") != sha_file(FREEZE):
        raise ScoreError("Prelabel verification is not bound to this freeze")
    if freeze.get("source_commit") not in subprocess_run_git_rev_list().splitlines():
        raise ScoreError("Frozen prediction source commit is not an ancestor of HEAD")
    if not is_committed_and_clean(FREEZE):
        raise ScoreError("Freeze must be committed and unchanged before target access")
    if not is_committed_and_clean(PRELABEL):
        raise ScoreError("Prelabel verification must be committed and unchanged before target access")
    train_hash = sha_file(TRAIN)
    expected_train_hash = freeze.get("input_hashes", {}).get("OFFICIAL::data/raw/classification/Train.csv")
    if expected_train_hash != train_hash:
        raise ScoreError("Official Train input hash differs from the frozen source")
    verified_bundles: dict[str, dict[str, str]] = {}
    for origin in (67, 75):
        for arm in ("REAL", "SHUFFLED"):
            unit_id = f"E{origin}_{arm}"
            path = RUN_ROOT / f"{unit_id}/predictions.npz"
            receipt_path = RUN_ROOT / f"{unit_id}/receipt.json"
            receipt = load_json(receipt_path)
            if not path.is_file() or receipt.get("artifact_sha256") != sha_file(path):
                raise ScoreError(f"Durable candidate output/receipt missing for {unit_id}")
            _assert_prelabel_artifact_hash(prelabel, unit_id, path)
            prelabel_receipt = prelabel.get("four_unit_receipts_and_outputs", {}).get(unit_id, {}).get("receipt_sha256")
            if not isinstance(prelabel_receipt, str) or sha_file(receipt_path) != prelabel_receipt:
                raise ScoreError(f"Current receipt is not the independently verified receipt: {unit_id}")
            verified_bundles[unit_id] = {"prediction_sha256": sha_file(path),
                                         "receipt_sha256": sha_file(receipt_path)}
    attempt = {
        "pilot_id": "CLS_A02_POSITION_KERNEL_V1",
        "status": "AUTHORIZED_SCORE_STARTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prelabel_verification_sha256": sha_file(PRELABEL),
        "freeze_sha256": sha_file(FREEZE),
        "source_commit": freeze["source_commit"],
        "official_train_sha256": train_hash,
        "candidate_bundles": verified_bundles,
        "target_source": "official Classification Train.csv artifact_signal, addressed only after the prelabel report is committed",
        "query_target_accessed_before_this_attempt": "known incident: E67 cached labels member was materialized during metadata inventory; values were not displayed or used",
        "scoring_runs_allowed": 1,
        "test_or_sample_submission_accessed": False,
        "kaggle_accessed": False,
    }
    with SCORE_ATTEMPT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(attempt, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    # This is the one authorized query-target access for scoring, after four
    # durable bundles and committed independent prelabel verification.
    train = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "artifact_signal"],
                        dtype={"record_id": "string"})
    if sha_file(TRAIN) != train_hash:
        raise ScoreError("Official Train changed while scoring target labels were read")
    e67r, e67s = load_unit(67, "REAL"), load_unit(67, "SHUFFLED")
    e75r, e75s = load_unit(75, "REAL"), load_unit(75, "SHUFFLED")
    y67r, y67s = _targets_for(e67r, train), _targets_for(e67s, train)
    y75r, y75s = _targets_for(e75r, train), _targets_for(e75s, train)
    if not np.array_equal(y67r, y67s) or not np.array_equal(y75r, y75s):
        raise ScoreError("REAL/SHUFFLED query target bindings differ")
    for real, shuffled, y in ((e67r, e67s, y67r), (e75r, e75s, y75r)):
        if not np.array_equal(real["missing_mask"], shuffled["missing_mask"]):
            raise ScoreError("REAL/SHUFFLED availability differs")
        if not np.array_equal(real["champion_class"], shuffled["champion_class"]):
            raise ScoreError("Full champion class replay differs by arm")
    if not np.array_equal(e67r["score_record_ids"][(e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79)].astype(str),
                          e75r["score_record_ids"][(e75r["score_cycles"] >= 76) & (e75r["score_cycles"] <= 79)].astype(str)):
        raise ScoreError("E67/E75 W6 order and IDs differ; cannot bind unique pool")
    if not np.array_equal(y67r[(e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79)],
                          y75r[(e75r["score_cycles"] >= 76) & (e75r["score_cycles"] <= 79)]):
        raise ScoreError("E67/E75 overlapping W6 target labels differ")

    unique_real, y_unique = _concat_unique(e67r, e75r, y67r, y75r)
    unique_shuffled, y_unique_s = _concat_unique(e67s, e75s, y67s, y75s)
    if not np.array_equal(y_unique, y_unique_s):
        raise ScoreError("Unique primary real/shuffled target rows differ")

    views: dict[str, Any] = {
        "UNIQUE_PRIMARY": _view_from_arrays("UNIQUE_PRIMARY", unique_real, unique_shuffled, y_unique,
                                             np.ones(len(y_unique), dtype=bool)),
        "E67_H12": _view_from_arrays("E67_H12", e67r, e67s, y67r,
                                      (e67r["score_cycles"] >= 68) & (e67r["score_cycles"] <= 79)),
        "E67_W4": _view_from_arrays("E67_W4", e67r, e67s, y67r,
                                     (e67r["score_cycles"] >= 68) & (e67r["score_cycles"] <= 71)),
        "E67_W5": _view_from_arrays("E67_W5", e67r, e67s, y67r,
                                     (e67r["score_cycles"] >= 72) & (e67r["score_cycles"] <= 75)),
        "E67_W6": _view_from_arrays("E67_W6", e67r, e67s, y67r,
                                     (e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79)),
        "E75_W6": _view_from_arrays("E75_W6", e75r, e75s, y75r,
                                     (e75r["score_cycles"] >= 76) & (e75r["score_cycles"] <= 79)),
    }
    subgroup_results = {
        "UNIQUE_PRIMARY": _subgroups(unique_real, y_unique, "UNIQUE_PRIMARY"),
        "E67_H12": _subgroups(e67r, y67r, "E67_H12", (e67r["score_cycles"] >= 68) & (e67r["score_cycles"] <= 79)),
        "E67_W4": _subgroups(e67r, y67r, "E67_W4", (e67r["score_cycles"] >= 68) & (e67r["score_cycles"] <= 71)),
        "E67_W5": _subgroups(e67r, y67r, "E67_W5", (e67r["score_cycles"] >= 72) & (e67r["score_cycles"] <= 75)),
        "E67_W6": _subgroups(e67r, y67r, "E67_W6", (e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79)),
        "E75_W6": _subgroups(e75r, y75r, "E75_W6", (e75r["score_cycles"] >= 76) & (e75r["score_cycles"] <= 79)),
    }
    subgroup_cells = [cell for view in subgroup_results.values() for family in view.values()
                      for cell in family.values()]
    supported = [cell for cell in subgroup_cells if cell["support"] == "SUPPORTED"]
    failed = [cell for cell in supported if not cell["guard_pass"]]
    subgroup_guard_pass = bool(supported) and not failed

    unique_gain = views["UNIQUE_PRIMARY"]["gain_real_vs_champion_macro_f1"]
    e67_gain = views["E67_H12"]["gain_real_vs_champion_macro_f1"]
    e75_gain = views["E75_W6"]["gain_real_vs_champion_macro_f1"]
    real_shuffle_unique = views["UNIQUE_PRIMARY"]["gain_real_vs_shuffled_macro_f1"]
    real_shuffle_e75 = views["E75_W6"]["gain_real_vs_shuffled_macro_f1"]
    scientific_gates = {
        "real_unique_pool_gain_ge_0_0005": unique_gain >= 0.0005,
        "real_e67_h12_gain_nonnegative": e67_gain >= 0.0,
        "real_e75_w6_gain_nonnegative": e75_gain >= 0.0,
        "real_vs_shuffled_unique_gain_ge_0_0002": real_shuffle_unique >= 0.0002,
        "real_vs_shuffled_e75_w6_gain_nonnegative": real_shuffle_e75 >= 0.0,
        "supported_legacy_guards_pass": subgroup_guard_pass,
    }
    exploratory_pool_policy = (unique_gain > 0.0) or (e75_gain >= 0.0005 and unique_gain >= -0.0005)
    missing_invariant = all(
        np.array_equal(bundle["p_a02_candidate"][bundle["missing_mask"]],
                       bundle["p_a02_original"][bundle["missing_mask"]])
        for bundle in (e67r, e67s, e75r, e75s)
    )
    exploratory_ok = exploratory_pool_policy and subgroup_guard_pass and missing_invariant \
        and prelabel.get("status") == "PASS_AMENDED_PRELABEL_VERIFICATION" \
        and prelabel.get("model_construction_training_boundary_verified") is True
    exploratory_status = "ELIGIBLE_EXPLORATORY_ONLY" if exploratory_ok else "NOT_ELIGIBLE_EXPLORATORY"

    per_unit = {}
    for origin, real, shuffled in ((67, e67r, e67s), (75, e75r, e75s)):
        for arm, bundle in (("REAL", real), ("SHUFFLED", shuffled)):
            per_unit[f"E{origin}_{arm}"] = {
                "rows": int(len(bundle["score_rows"])),
                "missing_rows": int(bundle["missing_mask"].sum()),
                "present_supported_rows": int(np.sum((~bundle["missing_mask"]) & (bundle["m0"] > 0))),
                "mean_neighbor_overlap_supported": float(np.mean(bundle["neighbor_overlap_fraction"][bundle["neighbor_count"] > 0])) if np.any(bundle["neighbor_count"] > 0) else 0.0,
                "correction_clip_rows": int(np.sum(np.abs(bundle["correction_raw"]) > 0.05)),
                "correction_clip_rate_supported_present": float(
                    np.sum((np.abs(bundle["correction_raw"]) > 0.05)
                           & ~bundle["missing_mask"] & (bundle["m0"] > 0))
                    / max(1, np.sum(~bundle["missing_mask"] & (bundle["m0"] > 0)))
                ),
                "max_abs_mass_error": float(np.max(np.abs(bundle["mass_error"]))),
            }

    cycle_signs = {
        "E67_REAL_MINUS_CHAMPION": _cycle_signs(e67r, y67r),
        "E67_SHUFFLED_MINUS_CHAMPION": _cycle_signs(e67s, y67s),
        "E75_REAL_MINUS_CHAMPION": _cycle_signs(e75r, y75r),
        "E75_SHUFFLED_MINUS_CHAMPION": _cycle_signs(e75s, y75s),
        "UNIQUE_REAL_MINUS_CHAMPION": _cycle_signs(unique_real, y_unique),
    }
    result = {
        "status": "SCORED_EXPLORATORY_WITH_PROTOCOL_DEVIATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_id": "CLS_A02_POSITION_KERNEL_V1",
        "source_commit": freeze["source_commit"],
        "official_source_commit": freeze["official_source_commit"],
        "official_train_sha256": train_hash,
        "candidate_bundles": verified_bundles,
        "freeze_sha256": sha_file(FREEZE),
        "prelabel_verification_sha256": sha_file(PRELABEL),
        "scoring_attempt_sha256": sha_file(SCORE_ATTEMPT),
        "target_source": "Official Classification Train.csv artifact_signal, selected by verified source row positions after independent prelabel verification was committed",
        "target_access": {"read_for_scoring": True, "read_after_prelabel_verification": True,
                          "read_before_prelabel_during_this_scoring_script": False,
                          "prior_incident": "E67 cached labels member materialized during metadata inventory; values not printed, summarized or used; protocol deviation preserved"},
        "original_prelabel_boundary_compliant": False,
        "scientific_qualification": False,
        "scientific_performance_gates": scientific_gates,
        "exploratory_eligibility": {
            "status": exploratory_status,
            "pool_policy_pass": bool(exploratory_pool_policy),
            "supported_group_harm_policy_pass": bool(subgroup_guard_pass),
            "missing_invariant_pass": bool(missing_invariant),
            "independent_prelabel_qa_pass": prelabel.get("status") == "PASS_AMENDED_PRELABEL_VERIFICATION",
            "training_boundary_pass": prelabel.get("model_construction_training_boundary_verified") is True,
            "scientific_qualification": False,
            "rule_source": "Unmodified ROOT_EXECUTION_PLAN; exploratory-only under ROOT incident amendment",
        },
        "views": views,
        "subgroups": subgroup_results,
        "subgroup_guard_summary": {"supported_cells": len(supported), "failed_cells": len(failed),
                                   "unknown_cells": len(subgroup_cells) - len(supported),
                                   "pass": bool(subgroup_guard_pass)},
        "per_unit_diagnostics": per_unit,
        "cycle_signs": cycle_signs,
        "overlap": {"unique_primary_rows": 20903,
                    "e67_e75_w6_rows": int(np.sum((e67r["score_cycles"] >= 76) & (e67r["score_cycles"] <= 79))),
                    "e67_e75_w6_counted_once_in_unique_pool": True},
        "limits": [
            "The E67 query labels member was materialized prematurely during metadata inventory; this run is not outcome-blind.",
            "The same development labels have been reused; views are adaptive temporal development evidence, not an untouched holdout.",
            "The operational missing-stream threshold was learned from global OOF; it is not purely past-only calibration.",
            "No Test/SampleSubmission/Kaggle access; no E79 construction or submission.",
            "Numerical performance gates and exploratory eligibility do not change the historical qualification flag.",
        ],
    }
    with RESULTS.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    with RESULTS_MD.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_md(result))
        stream.flush()
        os.fsync(stream.fileno())
    return result


def subprocess_run_git_rev_list() -> str:
    import subprocess
    return subprocess.run(["git", "rev-list", "HEAD"], cwd=REPO_ROOT,
                          check=True, text=True, capture_output=True).stdout


def is_tracked(path: Path) -> bool:
    import subprocess
    rel = path.relative_to(REPO_ROOT).as_posix()
    return subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                          text=True, capture_output=True).returncode == 0


def is_committed_and_clean(path: Path) -> bool:
    if not is_tracked(path):
        return False
    rel = path.relative_to(REPO_ROOT).as_posix()
    return subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT,
                          text=True, capture_output=True).returncode == 0


if __name__ == "__main__":
    try:
        print(json.dumps(score_once(), ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
