"""Additive one-shot recovery scorer for the frozen CLS_DAN_V1 pilot.

The original runner, frozen predictions, prelabel NPZ and R1 failure receipt
remain untouched. This module adds only an in-memory coverage alignment and
two separately reported E75 scientific gate interpretations authorized for R2.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cls_dan_v1 as runner
from scripts import cls_dan_v1_postfit_repair_r1 as repair
from scripts import verify_cls_dan_v1 as independent

REPORT_DIR = runner.REPORT_ROOT
AUTHORITY_PATH = REPORT_DIR / "ROOT_EVALUATION_RECOVERY_R2.json"
ACK_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_GATE_ACK.json"
INTENT_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_INTENT.json"
CONSUMED_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_CONSUMED.json"
STARTED_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_STARTED.json"
RESULTS_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_RESULTS.json"
VERIFY_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_INDEPENDENT_VERIFICATION.json"
COMPLETED_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_COMPLETED.json"
FAILURE_PATH = REPORT_DIR / "EVALUATION_RECOVERY_R2_FAILURE.json"
SCRIPT_REL = "scripts/cls_dan_v1_evaluation_recovery_r2.py"
TEST_REL = "tests/test_cls_dan_v1_evaluation_recovery_r2.py"
ACK_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_GATE_ACK.json"
INTENT_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_INTENT.json"
CONSUMED_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_CONSUMED.json"
STARTED_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_STARTED.json"
RESULTS_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_RESULTS.json"
VERIFY_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_INDEPENDENT_VERIFICATION.json"
COMPLETED_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_COMPLETED.json"
FAILURE_REL = "reports/simplification_v1/CLS_DAN_V1/EVALUATION_RECOVERY_R2_FAILURE.json"

VIEW_ROWS = {
    "E67_W4": 4950,
    "E67_W5": 7462,
    "E67_W6": 8491,
    "E67_H12": 20903,
    "E75_W6": 8491,
    "PRIMARY_UNIQUE": 20903,
}
PRELABEL_FIELDS = (
    "score_rows", "score_record_ids", "score_cycles", "present", "champion",
    "p_pos_original", "p_a02", "entry_friction", "text_load_group", "coverage",
    "dan", "linear", "aux_dan", "aux_linear",
)
R2_OUTPUTS = (CONSUMED_PATH, STARTED_PATH, RESULTS_PATH, VERIFY_PATH,
              COMPLETED_PATH, FAILURE_PATH)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_sha256(value: bytes) -> str:
    return hashlib.sha256(value.replace(b"\r\n", b"\n")).hexdigest()


def committed_bytes(relative: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])


def require_committed(path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = committed_bytes(relative)
    current = path.read_bytes()
    require(normalized_sha256(committed) == normalized_sha256(current),
            f"Required R2 evidence is not committed exactly: {relative}")
    return json.loads(current.decode("utf-8"))


def committed_source_hash(relative: str) -> str:
    disk = (ROOT / relative).read_bytes()
    blob = committed_bytes(relative)
    require(normalized_sha256(blob) == normalized_sha256(disk),
            f"R2 source file is not committed exactly: {relative}")
    return normalized_sha256(blob)


def assert_clean_checkout() -> str:
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True
    )
    require(not status.strip(), "R2 execution requires a clean tracked working tree")
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def expand_present_only_coverage(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return a copy with present-only coverage expanded to all rows; missing=0."""
    output = {key: np.asarray(value).copy() for key, value in record.items()}
    present = np.asarray(output["present"], dtype=bool).reshape(-1)
    coverage = np.asarray(output["coverage"], dtype=np.float64).reshape(-1)
    require(len(coverage) == int(present.sum()),
            "Coverage must be present-stream-only and align exactly to the present mask")
    require(np.isfinite(coverage).all() and np.all((coverage >= 0.0) & (coverage <= 1.0)),
            "Present-stream coverage contains nonfinite or out-of-range values")
    expanded = np.zeros(len(present), dtype=np.float64)
    expanded[present] = coverage
    require(np.all(expanded[~present] == 0.0),
            "Missing-stream coverage must be zero in the in-memory view")
    output["coverage"] = expanded
    return output


def load_prelabel(path: Path) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    data: dict[int, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any]
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["prelabel_metadata_json"].item()))
        for origin in runner.ORIGINS:
            prefix = f"E{origin}_"
            current = {key: np.asarray(archive[prefix + key]).copy()
                       for key in PRELABEL_FIELDS}
            current["score_rows"] = np.asarray(current["score_rows"], dtype=np.int64)
            current["score_record_ids"] = np.asarray(current["score_record_ids"]).astype(str)
            current["score_cycles"] = np.asarray(current["score_cycles"], dtype=np.int64)
            current["present"] = np.asarray(current["present"], dtype=bool)
            for key in ("champion", "p_pos_original", "p_a02", "dan", "linear",
                        "aux_dan", "aux_linear"):
                current[key] = np.asarray(current[key], dtype=np.float64)
            for key in ("entry_friction", "text_load_group"):
                current[key] = np.asarray(current[key]).astype(str)
            current["coverage"] = np.asarray(current["coverage"], dtype=np.float64)
            data[origin] = expand_present_only_coverage(current)
    return data, metadata


def validate_prelabel_shapes(data: Mapping[int, Mapping[str, Any]],
                             prediction_ready: Mapping[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for origin, expected_rows, expected_present in ((67, 20903, 19528), (75, 8491, 7965)):
        record = data[origin]
        lengths = {key: len(np.asarray(record[key]).reshape(-1)) for key in PRELABEL_FIELDS}
        require(set(lengths.values()) == {expected_rows},
                f"E{origin} expanded prelabel fields do not share the exact full-row shape")
        require(int(np.asarray(record["present"]).sum()) == expected_present,
                f"E{origin} prelabel availability count differs from the frozen receipt")
        require(np.all(np.asarray(record["coverage"])[~np.asarray(record["present"], dtype=bool)] == 0.0),
                f"E{origin} missing-stream expanded coverage is not zero")
        ids = np.asarray(record["score_record_ids"]).astype(str)
        rows = np.asarray(record["score_rows"], dtype=np.int64)
        cycles = np.asarray(record["score_cycles"], dtype=np.int64)
        expected = prediction_ready["origins"][str(origin)]
        require(len(np.unique(ids)) == expected_rows and len(np.unique(rows)) == expected_rows,
                f"E{origin} query IDs and source row positions must be unique")
        require(runner.sha256_array(ids) == expected["record_ids_sha256"]
                and runner.sha256_array(rows) == expected["row_positions_sha256"]
                and runner.sha256_array(cycles) == expected["cycles_sha256"],
                f"E{origin} query identity/cycle arrays differ from the committed readiness receipt")
        require(np.isfinite(np.asarray(record["champion"])).all()
                and np.isfinite(np.asarray(record["dan"])).all()
                and np.isfinite(np.asarray(record["linear"])).all(),
                f"E{origin} probability arrays contain nonfinite values")
        for model in ("champion", "dan", "linear"):
            probability = np.asarray(record[model], dtype=np.float64)
            require(np.all((probability >= 0.0) & (probability <= 1.0)),
                    f"E{origin} {model} probabilities are outside [0,1]")
            require(np.array_equal(probability[~record["present"]].view(np.uint64),
                                   np.asarray(record["champion"])[~record["present"]].view(np.uint64)),
                    f"E{origin} {model} missing fallback is not exact")
        audit[str(origin)] = {
            "rows": expected_rows,
            "present_rows": int(record["present"].sum()),
            "missing_rows": int((~record["present"]).sum()),
            "coverage_present_input_rows": expected_present,
            "coverage_full_aligned_rows": len(record["coverage"]),
            "coverage_missing_value": 0.0,
            "record_ids_sha256": runner.sha256_array(ids),
            "row_positions_sha256": runner.sha256_array(rows),
            "cycles_sha256": runner.sha256_array(cycles),
            "probabilities_finite_in_range": True,
            "missing_fallback_exact": True,
        }
    overlap = repair.validate_origin_alignment(data[67], data[75])
    require(overlap["shared_query_rows"] == 8491
            and overlap["source_rows_exact"] and overlap["cycles_exact"]
            and overlap["availability_exact"]
            and overlap["champion_probabilities_different_rows"] == 8491,
            "E67/E75 overlapping origins fail the frozen identity/origin-specific prediction check")
    views = runner._build_views(data)
    observed_rows = {name: int(len(view["record_ids"])) for name, view in views.items()}
    require(observed_rows == VIEW_ROWS,
            f"Frozen view row counts differ: expected {VIEW_ROWS}, got {observed_rows}")
    return {"origins": audit, "overlap": overlap, "view_rows": observed_rows,
            "coverage_alignment": "present-only to full rows in memory; missing=0; source NPZ unchanged"}


def validate_r2_authority(authority: Mapping[str, Any], source_freeze: Mapping[str, Any]) -> dict[str, Any]:
    require(authority.get("status") == "APPROVED_ONE_SAME_INPUT_EVALUATION_RECOVERY"
            and authority.get("candidate_id") == "CLS_DAN_V1"
            and authority.get("failed_attempt_commit") == "76d060c122b21302c11628201e833f0fed20c936"
            and authority.get("freeze_signature") == source_freeze.get("freeze_signature")
            and authority.get("original_source_commit") == source_freeze.get("source_commit"),
            "Committed root R2 authority does not match this frozen candidate/failed attempt")
    require(authority.get("original_evaluation_target_vector_reads") == 1
            and authority.get("original_evaluation_target_vector_rows") == 20903
            and authority.get("additional_selected_target_vector_reads_allowed") == 1
            and authority.get("expected_total_actual_evaluation_vector_reads") == 2
            and authority.get("new_predictor_or_representation_fits_allowed") == 0
            and authority.get("prediction_or_model_changes_allowed") is False,
            "Root R2 read/fit/prediction allowance differs from the exact authorized recovery")
    paths = {
        "prelabel_npz_sha256": runner.PRELABEL_PATH,
        "prediction_readiness_sha256": runner.PREDICTIONS_READY_PATH,
        "original_evaluation_arm_sha256": runner.RUN_ROOT / "EVALUATION_ARM.json",
        "original_evaluation_started_sha256": runner.RUN_ROOT / "EVALUATION_STARTED.json",
        "original_evaluation_consumed_sha256": runner.RUN_ROOT / "EVALUATION_CONSUMED.json",
        "original_evaluation_failure_sha256": REPORT_DIR / "EVALUATION_FAILURE_R1.json",
        "fit_ledger_sha256": runner.FIT_LEDGER_PATH,
    }
    for key, path in paths.items():
        require(path.is_file() and sha256_file(path) == authority.get(key),
                f"R2 root authority input hash mismatch: {key}")
        if path == runner.FIT_LEDGER_PATH:
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            require(normalized_sha256(committed_bytes(relative)) == normalized_sha256(path.read_bytes()),
                    "Fit ledger content is not committed exactly")
        elif path != runner.PRELABEL_PATH:
            require_committed(path)
    failure = require_committed(REPORT_DIR / "EVALUATION_FAILURE_R1.json")
    require(failure.get("status") == "EVALUATION_ATTEMPT_CONSUMED_NO_METRICS",
            "Original R1 evaluation failure receipt has an unexpected status")
    return {"verified": True, "bound_inputs": {key: authority[key] for key in paths},
            "prior_query_vector_reads": 1, "additional_read_authorized": 1,
            "expected_total_query_vector_reads": 2, "new_fits_allowed": 0,
            "predictions_may_change": False}


def audit_actual_prelabel(official_root: Path) -> dict[str, Any]:
    """Target-free source, receipt, and actual prelabel validation before intent."""
    authority = require_committed(AUTHORITY_PATH)
    source_freeze, _, _ = runner.verify_root_prefit_and_reservation(official_root)
    authority_audit = validate_r2_authority(authority, source_freeze)
    _, root_repair_authorization = repair._validate_root_repair(official_root)
    prediction_ready = require_committed(runner.PREDICTIONS_READY_PATH)
    features_ready = require_committed(runner.FEATURES_READY_PATH)
    postfit_receipt = require_committed(repair.POSTFIT_RECEIPT_PATH)
    binding = repair.validate_prelabel_binding(
        runner.PRELABEL_PATH, prediction_ready, postfit_receipt,
        expected_ready_sha256=postfit_receipt["prediction_ready_sha256"],
        enforce_git_policy=True,
    )
    data, metadata = load_prelabel(runner.PRELABEL_PATH)
    require(metadata.get("evaluation_query_targets_read") is False
            and metadata.get("test_opened") is False
            and metadata.get("sample_submission_opened") is False
            and metadata.get("kaggle_contacted") is False,
            "Prelabel metadata is not target-free")
    caches = {origin: repair._origin_cache(features_ready, origin) for origin in runner.ORIGINS}
    cache_audit = repair._verify_prelabel_against_caches(
        runner.PRELABEL_PATH, features_ready, caches,
        metadata["fit_receipt_hashes"], metadata["fit_artifact_hashes"],
    )
    require(cache_audit["target_free_payload_metadata"] is True,
            "Prelabel cache reconciliation did not confirm target-free provenance")
    require(postfit_receipt.get("saved_models_and_predictions_unchanged") is True,
            "R1 receipt does not certify unchanged model/prediction artifacts")
    require(all(value.get("prediction_readback_bit_identical") is True
                and value.get("prelabel_auxiliary_bit_identical") is True
                and value.get("query_labels_read") is False
                for value in postfit_receipt["saved_model_prediction_readback"].values()),
            "Saved prediction readback receipt is incomplete or altered")
    shape_audit = validate_prelabel_shapes(data, prediction_ready)
    return {
        "authority": authority,
        "authority_audit": authority_audit,
        "freeze": source_freeze,
        "root_repair_authorization": root_repair_authorization,
        "prediction_ready": prediction_ready,
        "features_ready": features_ready,
        "postfit_receipt": postfit_receipt,
        "binding": binding,
        "shape_audit": shape_audit,
        "cache_audit": cache_audit,
        "data": data,
        "prelabel_metadata": metadata,
    }


def gate_verdicts(views: Mapping[str, Mapping[str, Any]],
                  subgroup_rows: list[dict[str, Any]], qa_pass: bool) -> dict[str, Any]:
    """Return the original frozen scientific gate and separately amended R2 gate."""
    primary_gain = float(views["PRIMARY_UNIQUE"]["dan_gain_vs_champion"])
    e67_gain = float(views["E67_H12"]["dan_gain_vs_champion"])
    e75_gain = float(views["E75_W6"]["dan_gain_vs_champion"])
    primary_matched = float(views["PRIMARY_UNIQUE"]["dan_minus_linear"])
    e75_linear_gain = float(views["E75_W6"]["linear_gain_vs_champion"])
    e75_dan_minus_linear = float(views["E75_W6"]["dan_minus_linear"])
    supported = [row for row in subgroup_rows if row["status"] == "SUPPORTED"]
    all_views_supported = all(any(row["view"] == view and row["status"] == "SUPPORTED"
                                  for row in subgroup_rows)
                              for view in VIEW_ROWS)
    subgroup_scientific = all_views_supported and all(
        row["dan_macro_f1_harm_vs_champion"] <= 0.01
        and row["dan_positive_recall_harm_vs_champion"] <= 0.02 for row in supported
    )
    subgroup_exploratory = all_views_supported and all(
        row["dan_macro_f1_harm_vs_champion"] <= 0.02
        and row["dan_positive_recall_harm_vs_champion"] <= 0.03 for row in supported
    )
    shared = {
        "primary_gain_vs_champion_at_least_0_0005": primary_gain >= 0.0005,
        "E67_H12_gain_nonnegative": e67_gain >= 0.0,
        "E75_W6_gain_nonnegative": e75_gain >= 0.0,
        "primary_DAN_minus_LINEAR_at_least_0_0002": primary_matched >= 0.0002,
        "supported_all_view_subgroups_within_0_01_f1_0_02_recall": subgroup_scientific,
        "identity_route_probability_and_metric_QA": qa_pass,
    }
    original = {
        **shared,
        "E75_LINEAR_control_gain_nonnegative": e75_linear_gain >= 0.0,
    }
    amended = {
        **shared,
        "E75_DAN_minus_LINEAR_nonnegative_R2_amendment": e75_dan_minus_linear >= 0.0,
    }
    exploratory_performance = ((primary_gain > 0.0 and e75_gain >= -0.0005)
                               or (e75_gain >= 0.0005 and primary_gain >= -0.0005))
    exploratory = {
        "campaign_primary_or_E75_performance_rule": exploratory_performance,
        "supported_all_view_subgroups_within_0_02_f1_0_03_recall": subgroup_exploratory,
        "identity_route_probability_and_metric_QA": qa_pass,
    }
    return {
        "scientific_original_frozen": {
            "status": "PASS" if all(original.values()) else "FAIL",
            "E75_component": "LINEAR_macro_f1_minus_champion_macro_f1 >= 0",
            "components": original,
        },
        "scientific_r2_amended": {
            "status": "PASS" if all(amended.values()) else "FAIL",
            "E75_component": "DAN_macro_f1_minus_LINEAR_macro_f1 >= 0",
            "components": amended,
            "status_scope": "R2 diagnostic adjudication only; does not replace original verdict",
        },
        "exploratory_eligibility_original_rule": {
            "status": "PASS" if all(exploratory.values()) else "FAIL",
            "components": exploratory,
            "status_scope": "Unchanged and operative for E79 review; not an automatic authorization",
        },
        "mechanism_contrast": {
            "status": "PASS" if primary_matched >= 0.0002 else "FAIL",
            "definition": "predeclared primary DAN-minus-LINEAR Macro F1 >= +0.0002",
        },
    }


def evaluate_prediction_arrays(data: Mapping[int, Mapping[str, Any]], y_e67: np.ndarray,
                               prediction_ready: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score all frozen views from a single in-memory query label vector."""
    views = runner._build_views(data)
    labels = np.asarray(y_e67, dtype=np.int8).reshape(-1)
    e67_ids = np.asarray(data[67]["score_record_ids"]).astype(str)
    require(len(labels) == len(e67_ids) == 20903 and np.isin(labels, (0, 1)).all(),
            "R2 query label vector must be exactly 20,903 binary E67 query values")
    y_by_id = dict(zip(e67_ids.tolist(), labels.tolist()))
    require(len(y_by_id) == 20903, "E67 query IDs must be unique for label mapping")
    view_labels = {
        name: np.asarray([y_by_id[value] for value in view["record_ids"]], dtype=np.int8)
        for name, view in views.items()
    }
    view_results: dict[str, Any] = {}
    verifier_input: dict[str, dict[str, Any]] = {}
    subgroup_rows: list[dict[str, Any]] = []
    for name, view in views.items():
        y = view_labels[name]
        metrics = {model: runner._metrics(y, np.asarray(view[model]), np.asarray(view["present"]))
                   for model in ("champion", "dan", "linear")}
        view_results[name] = {
            "n": len(y), "cycle_min": int(np.min(view["cycles"])),
            "cycle_max": int(np.max(view["cycles"])), "metrics": metrics,
            "dan_gain_vs_champion": metrics["dan"]["macro_f1"] - metrics["champion"]["macro_f1"],
            "linear_gain_vs_champion": metrics["linear"]["macro_f1"] - metrics["champion"]["macro_f1"],
            "dan_minus_linear": metrics["dan"]["macro_f1"] - metrics["linear"]["macro_f1"],
            "mean_token_coverage": float(np.mean(view["coverage"])) if len(view["coverage"]) else None,
        }
        for model in ("champion", "dan", "linear"):
            verifier_input[f"{name}::{model}"] = {
                "probability": np.asarray(view[model], dtype=np.float64),
                "present": np.asarray(view["present"], dtype=bool),
            }
        subgroup_rows.extend(runner._subgroup_results(name, view, y))
    independent_by_view: dict[str, Any] = {}
    for name, view in views.items():
        model_map = {model: verifier_input[f"{name}::{model}"]
                     for model in ("champion", "dan", "linear")}
        independent_by_view[name] = independent.verify_metric_map(view_labels[name], model_map)
    verification_pass = all(row["status"] == "PASS" for row in independent_by_view.values())
    supported_per_view = {
        name: any(row["view"] == name and row["status"] == "SUPPORTED" for row in subgroup_rows)
        for name in views
    }
    support_valid = all(supported_per_view.values())
    for row in subgroup_rows:
        if row["status"] != "SUPPORTED":
            continue
        support = row["support"]
        require(support["n"] >= 200 and support["positive"] >= 30 and support["negative"] >= 30,
                f"Supported subgroup guard failed for {row['view']}/{row['variable']}/{row['group']}")
    route_check = all(
        prediction_ready["origins"][str(origin)]["missing_route_exact_for_dan"]
        and prediction_ready["origins"][str(origin)]["missing_route_exact_for_linear"]
        for origin in runner.ORIGINS
    )
    unique_id_check = (len(np.unique(views["PRIMARY_UNIQUE"]["record_ids"])) == 20903
                       and len(views["PRIMARY_UNIQUE"]["record_ids"]) == 20903)
    probability_check = all(
        np.isfinite(np.asarray(views[name][model])).all()
        and np.all((np.asarray(views[name][model]) >= 0.0)
                   & (np.asarray(views[name][model]) <= 1.0))
        for name in views for model in ("champion", "dan", "linear")
    )
    qa_pass = route_check and unique_id_check and probability_check and verification_pass and support_valid
    gates = gate_verdicts(view_results, subgroup_rows, qa_pass)
    per_cycle: list[dict[str, Any]] = []
    for origin, record in data.items():
        cycles = np.asarray(record["score_cycles"], dtype=np.int64)
        for cycle in sorted(np.unique(cycles).tolist()):
            mask = cycles == cycle
            ids = np.asarray(record["score_record_ids"]).astype(str)[mask]
            y_cycle = np.asarray([y_by_id[value] for value in ids], dtype=np.int8)
            cycle_metrics = {model: runner._metrics(
                y_cycle, np.asarray(record[model])[mask], np.asarray(record["present"])[mask]
            ) for model in ("champion", "dan", "linear")}
            per_cycle.append({
                "origin": origin, "cycle": int(cycle), "n": int(mask.sum()),
                "dan_gain_vs_champion": cycle_metrics["dan"]["macro_f1"] - cycle_metrics["champion"]["macro_f1"],
                "dan_direction": "POSITIVE" if cycle_metrics["dan"]["macro_f1"] > cycle_metrics["champion"]["macro_f1"] else "NEGATIVE" if cycle_metrics["dan"]["macro_f1"] < cycle_metrics["champion"]["macro_f1"] else "TIE",
                "linear_gain_vs_champion": cycle_metrics["linear"]["macro_f1"] - cycle_metrics["champion"]["macro_f1"],
                "linear_direction": "POSITIVE" if cycle_metrics["linear"]["macro_f1"] > cycle_metrics["champion"]["macro_f1"] else "NEGATIVE" if cycle_metrics["linear"]["macro_f1"] < cycle_metrics["champion"]["macro_f1"] else "TIE",
            })
    result = {
        "candidate_id": "CLS_DAN_V1", "status": "PASS_EVALUATED" if qa_pass else "FAIL_EVALUATION_QA",
        "recovery_id": "R2", "freeze_signature": prediction_ready["freeze_signature"],
        "source_commit": prediction_ready["source_commit"],
        "query_target_read": {
            "status": "ONE_ADDITIONAL_FROZEN_QUERY_VECTOR_READ",
            "evaluation_rows": len(labels), "evaluation_label_values_read_once_this_recovery": len(labels),
            "positive": int(labels.sum()), "negative": int(len(labels) - labels.sum()),
            "row_positions_sha256": runner.sha256_array(data[67]["score_rows"]),
            "record_ids_sha256": runner.sha256_array(data[67]["score_record_ids"]),
            "label_vector_sha256": runner.sha256_array(labels),
            "campaign_total_actual_query_vector_reads": 2,
            "campaign_read_accounting": "One consumed R1 vector read failed before metrics; this is the single R2 recovery vector read.",
            "independent_metric_verifier_reused_same_in_memory_labels": True,
            "test_opened": False, "sample_submission_opened": False, "kaggle_contacted": False,
        },
        "primary_unique_construction": "E67 cycles 68-75 plus E75 cycles 76-79",
        "coverage_note": "Present-only coverage was expanded in memory to full row order with missing=0; original NPZ unchanged.",
        "views": view_results, "subgroups": subgroup_rows, "per_cycle_signs": per_cycle,
        "independent_verification": independent_by_view,
        "qa": {
            "route_exact": route_check, "unique_primary_ids": unique_id_check,
            "finite_probabilities": probability_check, "independent_metrics_match": verification_pass,
            "all_views_have_supported_subgroup": support_valid,
            "view_row_counts_exact": {name: int(len(view["record_ids"])) == VIEW_ROWS[name]
                                       for name, view in views.items()},
            "missing_coverage_zero": all(
                np.all(np.asarray(data[origin]["coverage"])[~np.asarray(data[origin]["present"], dtype=bool)] == 0.0)
                for origin in runner.ORIGINS
            ),
            "test_opened": False, "sample_submission_opened": False, "kaggle_contacted": False,
        },
        "gates": gates,
        "interpretation_limits": [
            "E67/E75 temporal views overlap and are development evidence, not independent fresh holdouts.",
            "E75 cycles 68-75 were in its fit population; primary unique predictions for cycles 68-75 use E67.",
            "The LINEAR control shares the learned E32 map but not DAN capacity/regularization paths; the contrast is not a causal isolation of nonlinearity.",
            "The learned unigram map does not establish token semantics.",
            "Missing-stream rows use the exact champion fallback; auxiliary predictors were fit only on present-stream rows.",
            "No Test, SampleSubmission, full-Train refit, or Kaggle access occurred.",
        ],
    }
    verification = {
        "candidate_id": "CLS_DAN_V1", "recovery_id": "R2",
        "status": "PASS" if verification_pass else "FAIL",
        "label_source": "same single R2 in-memory query vector",
        "views": independent_by_view,
    }
    return result, verification


def _target_free_hash_map(audit: Mapping[str, Any]) -> dict[str, str]:
    authority = audit["authority"]
    freeze = audit["freeze"]
    return {
        "root_authority": sha256_file(AUTHORITY_PATH),
        "source_input_freeze": sha256_file(runner.FREEZE_PATH),
        "source_input_signature": freeze["freeze_signature"],
        "prelabel_npz": audit["binding"]["prelabel_sha256"],
        "prediction_readiness": audit["binding"]["prediction_ready_sha256"],
        "postfit_prelabel_receipt": sha256_file(repair.POSTFIT_RECEIPT_PATH),
        "evaluation_arm": authority["original_evaluation_arm_sha256"],
        "evaluation_started_r1": authority["original_evaluation_started_sha256"],
        "evaluation_consumed_r1": authority["original_evaluation_consumed_sha256"],
        "evaluation_failure_r1": authority["original_evaluation_failure_sha256"],
        "fit_ledger": authority["fit_ledger_sha256"],
    }


def preflight(official_root: Path) -> dict[str, Any]:
    require(not any(path.exists() for path in (INTENT_PATH, *R2_OUTPUTS)),
            "R2 recovery evidence already exists; no preflight overwrite or retry")
    audit = audit_actual_prelabel(official_root)
    code_hashes = {SCRIPT_REL: normalized_sha256(Path(__file__).read_bytes()),
                   TEST_REL: normalized_sha256((ROOT / TEST_REL).read_bytes())}
    ack = require_committed(ACK_PATH)
    reconciliation = ack.get("specification_reconciliation", {})
    require(ack.get("status") == "ACKNOWLEDGED_BEFORE_R2_TARGET_READ"
            and reconciliation.get("original_frozen_E75_component") == "LINEAR_minus_champion >= 0"
            and reconciliation.get("R2_authority_amended_E75_component") == "DAN_minus_LINEAR >= 0"
            and reconciliation.get("exploratory_rule_changed") is False,
            "Committed R2 gate acknowledgment does not define the dual-verdict policy")
    report = {
        "status": "PASS_R2_PREACCESS_NO_QUERY_LABELS",
        "candidate_id": "CLS_DAN_V1", "recovery_id": "R2",
        "authority_path": AUTHORITY_PATH.relative_to(ROOT).as_posix(),
        "authority_sha256": sha256_file(AUTHORITY_PATH),
        "authority_audit": audit["authority_audit"],
        "source_commit": audit["freeze"]["source_commit"],
        "freeze_signature": audit["freeze"]["freeze_signature"],
        "official_train_sha256": audit["freeze"]["official_train_sha256"],
        "target_free_hashes": _target_free_hash_map(audit),
        "prelabel_shape_audit": audit["shape_audit"],
        "cache_reconciliation": audit["cache_audit"],
        "gate_acknowledgment_path": ACK_REL,
        "gate_acknowledgment_sha256": sha256_file(ACK_PATH),
        "gate_verdict_policy": {
            "original_frozen_scientific": "E75 LINEAR-minus-champion >= 0; retain as authoritative historical verdict",
            "R2_amended_scientific": "E75 DAN-minus-LINEAR >= 0; separate diagnostic adjudication only",
            "exploratory_eligibility": "unchanged original campaign rule, operative for E79 review",
            "outcome_blind": True,
        },
        "code_test_hashes": code_hashes,
        "expected_query_vector": {
            "rows": 20903,
            "record_ids_sha256": runner.sha256_array(audit["data"][67]["score_record_ids"]),
            "row_positions_sha256": runner.sha256_array(audit["data"][67]["score_rows"]),
            "one_additional_read_authorized": 1,
            "expected_campaign_total_actual_reads": 2,
        },
        "fits_or_prediction_changes": 0,
        "query_labels_read": False,
        "metrics_computed": False,
        "test_opened": False,
        "sample_submission_opened": False,
        "kaggle_contacted": False,
    }
    runner.atomic_json(INTENT_PATH, report, write_once=True)
    return report


def arm_recovery() -> dict[str, Any]:
    intent = require_committed(INTENT_PATH)
    ack = require_committed(ACK_PATH)
    require(intent["status"] == "PASS_R2_PREACCESS_NO_QUERY_LABELS"
            and intent["gate_acknowledgment_sha256"] == sha256_file(ACK_PATH)
            and ack["status"] == "ACKNOWLEDGED_BEFORE_R2_TARGET_READ",
            "R2 intent or gate acknowledgment is missing, changed, or not committed")
    for relative, expected in intent["code_test_hashes"].items():
        require(committed_source_hash(relative) == expected,
                f"R2 source/test hash changed after preflight: {relative}")
    require(not any(path.exists() for path in R2_OUTPUTS),
            "R2 attempt has already been consumed; no retry")
    current_head = assert_clean_checkout()
    consumed = {
        "status": "R2_ADDITIONAL_QUERY_VECTOR_READ_CONSUMED_NO_RETRY",
        "candidate_id": "CLS_DAN_V1", "recovery_id": "R2",
        "intent_sha256": sha256_file(INTENT_PATH),
        "root_authority_sha256": sha256_file(AUTHORITY_PATH),
        "prelabel_sha256": intent["target_free_hashes"]["prelabel_npz"],
        "previous_actual_query_vector_reads": 1,
        "r2_additional_query_vector_reads_authorized": 1,
        "expected_campaign_total_actual_query_vector_reads": 2,
        "query_rows": intent["expected_query_vector"]["rows"],
        "record_ids_sha256": intent["expected_query_vector"]["record_ids_sha256"],
        "row_positions_sha256": intent["expected_query_vector"]["row_positions_sha256"],
        "query_read_status": "PENDING_AFTER_COMMITTED_NO_RETRY_MARKER",
        "query_labels_read": False,
        "no_retry_after_marker": True,
    }
    runner.atomic_json(CONSUMED_PATH, consumed, write_once=True)
    started = {
        "status": "R2_READ_ATTEMPT_STARTED_NO_RETRY",
        "candidate_id": "CLS_DAN_V1", "recovery_id": "R2",
        "consumed_receipt_sha256": sha256_file(CONSUMED_PATH),
        "head_before_marker_commit": current_head,
        "query_vector_rows": 20903,
        "query_vector_read_status": "PENDING_AFTER_COMMITTED_START_MARKER",
        "no_retry_after_marker": True,
        "query_labels_read": False,
    }
    runner.atomic_json(STARTED_PATH, started, write_once=True)
    return {"consumed": consumed, "started": started,
            "required_commit": current_head}


def _validate_recovery_ready(official_root: Path) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    intent = require_committed(INTENT_PATH)
    consumed = require_committed(CONSUMED_PATH)
    started = require_committed(STARTED_PATH)
    current_head = assert_clean_checkout()
    marker_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", STARTED_REL], text=True
    ).strip()
    require(marker_commit == current_head,
            "R2 consumed/start markers must be in current HEAD before target access")
    require(consumed.get("status") == "R2_ADDITIONAL_QUERY_VECTOR_READ_CONSUMED_NO_RETRY"
            and started.get("consumed_receipt_sha256") == sha256_file(CONSUMED_PATH)
            and intent.get("expected_campaign_total_actual_query_vector_reads") == 2
            and not any(path.exists() for path in (RESULTS_PATH, VERIFY_PATH, COMPLETED_PATH, FAILURE_PATH)),
            "R2 read is already consumed or has prior results; no second target access")
    for relative, expected in intent["code_test_hashes"].items():
        require(committed_source_hash(relative) == expected,
                f"R2 source/test hash changed before target access: {relative}")
    audit = audit_actual_prelabel(official_root)
    require(audit["shape_audit"] == intent["prelabel_shape_audit"],
            "Actual prelabel shape/identity audit changed after R2 preflight")
    require(_target_free_hash_map(audit) == intent["target_free_hashes"],
            "R2 target-free source, fit ledger or prelabel hashes changed after preflight")
    return intent, audit["data"], audit["prediction_ready"]


def evaluate(official_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    intent, data, prediction_ready = _validate_recovery_ready(official_root)
    rows = np.asarray(data[67]["score_rows"], dtype=np.int64)
    record_ids = np.asarray(data[67]["score_record_ids"]).astype(str)
    require(len(rows) == len(record_ids) == 20903
            and runner.sha256_array(rows) == intent["expected_query_vector"]["row_positions_sha256"]
            and runner.sha256_array(record_ids) == intent["expected_query_vector"]["record_ids_sha256"],
            "Exact frozen R2 query row/ID vector differs from committed preflight")
    train_path = Path(official_root) / "data/raw/classification/Train.csv"
    read_calls = 0
    try:
        # The only authorized query-target access in this recovery is this call.
        read_calls += 1
        labels = runner._read_exact_targets(
            train_path, rows, record_ids, purpose="single authorized CLS_DAN_V1 R2 recovery vector",
        )
        require(read_calls == 1 and len(labels) == 20903,
                "R2 must make exactly one query-vector reader call with 20,903 values")
        result, verification = evaluate_prediction_arrays(data, labels, prediction_ready)
        result["recovery_receipts"] = {
            "r2_intent_sha256": sha256_file(INTENT_PATH),
            "r2_consumed_sha256": sha256_file(CONSUMED_PATH),
            "r2_started_sha256": sha256_file(STARTED_PATH),
            "r2_read_attempt_calls": read_calls,
            "previous_r1_query_vector_read": 1,
            "campaign_total_actual_query_vector_reads": 2,
        }
        result_json = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
        verification_json = json.dumps(verification, sort_keys=True, ensure_ascii=False, allow_nan=False)
        result = json.loads(result_json)
        verification = json.loads(verification_json)
        runner.atomic_json(RESULTS_PATH, result, write_once=True)
        runner.atomic_json(VERIFY_PATH, verification, write_once=True)
        _write_results_markdown(result, REPORT_DIR / "EVALUATION_RECOVERY_R2.md")
        completed = {
            "status": "PASS_R2_EVALUATION_COMPLETED" if result["status"] == "PASS_EVALUATED" else "FAIL_R2_EVALUATION_QA",
            "candidate_id": "CLS_DAN_V1", "recovery_id": "R2",
            "r2_query_vector_reads": 1,
            "previous_r1_failed_query_vector_reads": 1,
            "campaign_total_actual_query_vector_reads": 2,
            "query_vector_rows": 20903,
            "prelabel_sha256": intent["target_free_hashes"]["prelabel_npz"],
            "results_sha256": sha256_file(RESULTS_PATH),
            "independent_verification_sha256": sha256_file(VERIFY_PATH),
            "scientific_original_frozen": result["gates"]["scientific_original_frozen"]["status"],
            "scientific_r2_amended": result["gates"]["scientific_r2_amended"]["status"],
            "exploratory_eligibility_original_rule": result["gates"]["exploratory_eligibility_original_rule"]["status"],
            "query_labels_persisted": False,
            "test_opened": False, "sample_submission_opened": False, "kaggle_contacted": False,
        }
        runner.atomic_json(COMPLETED_PATH, completed, write_once=True)
        return result, completed
    except Exception as exc:
        # A started marker is permanently consumed even if scoring/serialization fails.
        if not FAILURE_PATH.exists():
            runner.atomic_json(FAILURE_PATH, {
                "status": "R2_ATTEMPT_FAILED_NO_RETRY",
                "candidate_id": "CLS_DAN_V1", "recovery_id": "R2",
                "failure_type": type(exc).__name__, "failure_message": str(exc),
                "reader_calls_entered": read_calls,
                "query_labels_persisted": False,
                "query_vector_reads_total_if_reader_entered": 2 if read_calls else 1,
                "no_retry": True,
            }, write_once=True)
        raise


def _write_results_markdown(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# CLS_DAN_V1 evaluation recovery R2", "",
        "One additional exact query-label vector read was used after the R2 markers were committed. The original consumed R1 attempt and failure receipt remain unchanged.", "",
        "## Main metrics", "",
        "| View | Rows | Champion Macro F1 | DAN Macro F1 | DAN gain | LINEAR Macro F1 | LINEAR gain | DAN − LINEAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("PRIMARY_UNIQUE", "E67_W4", "E67_W5", "E67_W6", "E67_H12", "E75_W6"):
        row = result["views"][name]
        m = row["metrics"]
        lines.append(
            f"| {name} | {row['n']} | {m['champion']['macro_f1']:.6f} | {m['dan']['macro_f1']:.6f} | "
            f"{row['dan_gain_vs_champion']:+.6f} | {m['linear']['macro_f1']:.6f} | "
            f"{row['linear_gain_vs_champion']:+.6f} | {row['dan_minus_linear']:+.6f} |"
        )
    lines.extend(["", "## Gate verdicts", ""])
    for name, verdict in result["gates"].items():
        lines.append(f"- **{name}: {verdict['status']}**")
    lines.extend([
        "", "The original frozen scientific verdict preserves E75 LINEAR-minus-champion >= 0. The separately reported R2-amended scientific verdict uses E75 DAN-minus-LINEAR >= 0 under the later root adjudication; it does not replace the original verdict.",
        "", "The exploratory eligibility rule is unchanged and remains the operative criterion for any E79 review. Neither scientific verdict authorizes more fitting, Test inference, or Kaggle action.",
        "", "## Limits", "",
        "E67/E75 temporal views overlap; E75 training includes cycles 68-75. The primary unique view uses E67 predictions for cycles 68-75 and E75 for cycles 76-79. The comparison does not isolate a causal nonlinear effect. Missing-stream rows use exact champion fallback.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _synthetic_arrays() -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    w4 = [1200, 1200, 1250, 1300]
    w5 = [1865, 1865, 1866, 1866]
    w6 = [2123, 2123, 2123, 2122]
    counts = w4 + w5 + w6
    cycles = np.concatenate([np.full(n, c, dtype=np.int64)
                             for c, n in zip(range(68, 80), counts)])
    rows = np.arange(len(cycles), dtype=np.int64)
    require(len(rows) == 20903 and int(np.sum((cycles >= 68) & (cycles <= 71))) == 4950
            and int(np.sum((cycles >= 72) & (cycles <= 75))) == 7462
            and int(np.sum(cycles >= 76)) == 8491,
            "Synthetic frozen-view sizes are invalid")
    ids = np.asarray([f"SYN{value:013d}" for value in rows], dtype=str)
    y = ((rows % 4) >= 2).astype(np.int8)
    present = rows % 11 != 0
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for origin, select, multiplier, offset, origin_delta in (
        (67, np.ones(len(rows), dtype=bool), 17, 0, 0.0),
        (75, cycles >= 76, 29, 7, 0.013),
    ):
        local_rows, local_ids, local_cycles = rows[select], ids[select], cycles[select]
        local_present = present[select]
        base = 0.12 + 0.76 * (((local_rows * multiplier + offset) % 97) / 96.0)
        champion = (base + origin_delta).astype(np.float64)
        dan_aux = np.clip(0.47 + 0.3 * np.sin(local_rows * 0.013), 0.02, 0.98)
        linear_aux = np.clip(0.51 + 0.3 * np.cos(local_rows * 0.019), 0.02, 0.98)
        dan, linear = champion.copy(), champion.copy()
        dan[local_present] = 0.75 * champion[local_present] + 0.25 * dan_aux[local_present]
        linear[local_present] = 0.75 * champion[local_present] + 0.25 * linear_aux[local_present]
        friction = np.where(local_rows < 20, "rare", np.where(local_rows % 2 == 0, "low", "high"))
        groups = np.where((local_rows // 3) % 2 == 0, "Q1", "Q2")
        outputs[origin] = {
            "score_rows": local_rows, "score_record_ids": local_ids,
            "score_cycles": local_cycles, "present": local_present,
            "champion": champion, "dan": dan, "linear": linear,
            "p_pos_original": np.clip(champion - 0.03, 0.0, 1.0),
            "p_a02": np.clip(champion + 0.02, 0.0, 1.0),
            "entry_friction": friction.astype(str), "text_load_group": groups.astype(str),
            "coverage": ((local_rows[local_present] % 100) / 100.0).astype(np.float64),
            "aux_dan": np.where(local_present, dan_aux, np.nan).astype(np.float64),
            "aux_linear": np.where(local_present, linear_aux, np.nan).astype(np.float64),
        }
    return outputs, y


def synthetic_cli(output_dir: Path) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not any(root.iterdir()), "Synthetic output directory must be empty; persistence guard prevents retry")
    data, labels = _synthetic_arrays()
    overlap = repair.validate_origin_alignment(data[67], data[75])
    require(overlap["shared_query_rows"] == 8491
            and overlap["champion_probabilities_different_rows"] == 8491,
            "Synthetic origins must share IDs but retain origin-specific champions")
    prelabel = root / "PRELABEL.npz"
    payload: dict[str, np.ndarray] = {}
    for origin, record in data.items():
        payload.update({f"E{origin}_{key}": value for key, value in record.items()})
    runner.atomic_npz(prelabel, **payload)
    train = root / "Train.csv"
    with train.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_id", "artifact_signal"])
        writer.writerows(zip(data[67]["score_record_ids"].tolist(), labels.tolist()))
    # The same persisted one-shot boundary is exercised, but labels are synthetic.
    consumed = root / "R2_CONSUMED.json"
    runner.atomic_json(consumed, {"status": "CONSUMED_NO_RETRY", "query_labels_read": False}, write_once=True)
    read_calls = 0
    read_calls += 1
    read_labels = runner._read_exact_targets(
        train, data[67]["score_rows"], data[67]["score_record_ids"],
        purpose="synthetic recovery CLI lifecycle",
    )
    prediction_ready = {
        "freeze_signature": "SYNTHETIC_ONLY", "source_commit": "synthetic",
        "origins": {str(origin): {"missing_route_exact_for_dan": True,
                                  "missing_route_exact_for_linear": True}
                    for origin in runner.ORIGINS},
    }
    expanded_data = {origin: expand_present_only_coverage(record)
                     for origin, record in data.items()}
    result, verification = evaluate_prediction_arrays(expanded_data, read_labels, prediction_ready)
    require(read_calls == 1 and len(read_labels) == 20903,
            "Synthetic CLI must read one exact 20,903-label vector")
    result["synthetic_qa"] = {
        "target_read_calls": read_calls, "actual_query_labels_read": False,
        "overlap": overlap, "prelabel_sha256": sha256_file(prelabel),
        "present_only_coverage_input": {str(o): int(len(data[o]["coverage"])) for o in runner.ORIGINS},
        "expanded_coverage_rows": {str(o): int(len(expanded_data[o]["coverage"])) for o in runner.ORIGINS},
    }
    result["query_target_read"]["campaign_total_actual_query_vector_reads"] = "NOT_APPLICABLE_SYNTHETIC_ONLY"
    result["query_target_read"]["campaign_read_accounting"] = "Synthetic QA only; no official query labels accessed."
    runner.atomic_json(root / "RESULTS.json", result, write_once=True)
    runner.atomic_json(root / "INDEPENDENT_VERIFICATION.json", verification, write_once=True)
    runner.atomic_json(root / "R2_COMPLETED.json", {
        "status": "PASS_SYNTHETIC_R2_CLI_QA", "target_read_calls": read_calls,
        "views": VIEW_ROWS, "result_sha256": sha256_file(root / "RESULTS.json"),
        "verification_sha256": sha256_file(root / "INDEPENDENT_VERIFICATION.json"),
        "no_real_target_access": True,
    }, write_once=True)
    return json.loads((root / "R2_COMPLETED.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLS_DAN_V1 additive R2 evaluation recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="Target-free R2 binding and shape audit")
    preflight_parser.add_argument("--official-root", type=Path, required=True)
    commands.add_parser("arm", help="Write one-shot no-retry markers; commit before evaluation")
    evaluate_parser = commands.add_parser("evaluate", help="Read one exact R2 query-label vector and score")
    evaluate_parser.add_argument("--official-root", type=Path, required=True)
    synthetic_parser = commands.add_parser("synthetic", help="Run full synthetic CLI/scorer/persistence QA")
    synthetic_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        print(json.dumps(synthetic_cli(args.output_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        print(json.dumps(preflight(args.official_root), indent=2, sort_keys=True))
        return 0
    if args.command == "arm":
        print(json.dumps(arm_recovery(), indent=2, sort_keys=True))
        return 0
    result, receipt = evaluate(args.official_root)
    print(json.dumps({"result_status": result["status"], "gates": result["gates"],
                      "completed": receipt}, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
