"""Root-authorized, one-read recovery for the failed CLS PPMI R1 scoring pass."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
_R1_RUNNER_PATH = SCRIPT_DIR / "cls_ppmi_context_correction_r1.py"
_R1_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "cls_ppmi_context_correction_r1_runner_for_r2", _R1_RUNNER_PATH)
if _R1_RUNNER_SPEC is None or _R1_RUNNER_SPEC.loader is None:
    raise RuntimeError("Could not load the committed R1 runner for the R2 adapter")
r1 = importlib.util.module_from_spec(_R1_RUNNER_SPEC)
sys.modules[_R1_RUNNER_SPEC.name] = r1
_R1_RUNNER_SPEC.loader.exec_module(r1)
from verify_cls_ppmi_context_recovery_r2 import (  # noqa: E402
    IndependentVerificationError,
    dual_e75_gate_readback,
    verify_in_memory_recovery_r2,
)

BASE_REPORT_DIR = r1.BASE_REPORT_DIR
BASE_EXPERIMENT_DIR = r1.BASE_EXPERIMENT_DIR
R1_REPORT_DIR = r1.CORRECTION_REPORT_DIR
R1_EXPERIMENT_DIR = r1.CORRECTION_EXPERIMENT_DIR
R2_REPORT_DIR = BASE_REPORT_DIR / "CORRECTION_R2"
R2_EXPERIMENT_DIR = BASE_EXPERIMENT_DIR / "CORRECTION_R2"
R1_FAILURE_PATH = R1_EXPERIMENT_DIR / "FAILED_VERIFICATION_R1.json"
RECOVERY_RECEIPT_PATH = R2_EXPERIMENT_DIR / "RECOVERY_AUTH_RECEIPT.json"
INTENT_PATH = R2_EXPERIMENT_DIR / "CORRECTION_INTENT_R2.json"
ATTEMPT_PATH = R2_EXPERIMENT_DIR / "SCORE_READ_ATTEMPT_R2.json"
RESULTS_PATH = R2_EXPERIMENT_DIR / "CORRECTED_RESULTS_R2.json"
REPORT_PATH = R2_EXPERIMENT_DIR / "CORRECTED_REPORT_R2.md"
VERIFICATION_PATH = R2_EXPERIMENT_DIR / "IN_MEMORY_VERIFICATION_R2.json"
RECEIPT_PATH = R2_EXPERIMENT_DIR / "CORRECTION_RECEIPT_R2.json"
SYNTHETIC_QA_PATH = R2_REPORT_DIR / "SYNTHETIC_QA_R2.json"
PROTOCOL_PATH = R2_REPORT_DIR / "PROTOCOL_R2.md"

R1_INTENT_PATH = R1_EXPERIMENT_DIR / "CORRECTION_INTENT.json"
R1_ATTEMPT_PATH = R1_EXPERIMENT_DIR / "SCORE_READ_ATTEMPT.json"
ROOT_CAMPAIGN_PROTOCOL_PATH = BASE_REPORT_DIR / "PROTOCOL.md"
ROOT_CAMPAIGN_RUNNER_PATH = SCRIPT_DIR / "cls_ppmi_context_v1.py"

ROOT_DECISION_THREAD_ID = "01a19569-16e0-7253-aa14-57a0a3160e03"
ROOT_DECISION_MESSAGE = """ROOT DECISION: APPROVE ONE BOUNDED PPMI SCORING RECOVERY R2, zero fits. I independently inspected the actual verifier: line351 wrongly indexes gain_real_vs_champion as a model, confirming your blocker. Preserve 38cae991, the consumed SCORE_READ_ATTEMPT and all old artifacts; record failed correction R1 accurately. Add a separate committed root-authorized recovery receipt binding this message, original freeze/prediction hashes, R1 intent+attempt, and corrected source hashes. Allow exactly ONE further selected target read of the SAME frozen 20,903 rows/IDs, only after source/test/intent commit. Total evaluation reads must truthfully be 3: original50,608 + failedR1 selected20,903 + R2 selected20,903 (92,414 materialized target values across evaluations); verifier uses in-memory labels only. No new fits, prediction/policy/threshold/support changes, Test/Kaggle access or automatic retries. Fix anchor handling explicitly: three model anchors map to their macro_f1; gain_real_vs_champion maps to the derived candidate_real_minus_champion_macro_f1. Reject unknown anchor keys. Run a complete synthetic scorer+verifier+serialization path with a nonempty anchor mapping having EXACT production schema, not anchors={}; cover mismatched anchor rejection, all6windows, allviewguards and read guard before production read. Minimal additive recovery adapter is fine; preserve frozen sources and originalfailed path. Important shared specification clarification: original ROOT_CAMPAIGN says E75controlgain>=0; do not claim REAL-minus-SHUFFLED is unambiguously the original frozen contract. Report original SHUFFLED-minus-champion gate and R1-amended REAL-minus-SHUFFLED gate SEPARATELY; preserve original verdict, label amended comparison later root amendment. Exploratory eligibility remains unchanged. Then execute the one authorized recovery and independent QA without another ceremonial approval; return results/QA/read accounting + one next mechanism recommendation. If another technical failure occurs, preserve checkpoint and return, no further read."""

SOURCE_PATHS = (
    "src/models/cls_ppmi_context_correction_r1.py",
    "scripts/cls_ppmi_context_correction_r1.py",
    "scripts/verify_cls_ppmi_context_correction_r1.py",
    "tests/test_cls_ppmi_context_correction_r1.py",
    "scripts/verify_cls_ppmi_context_recovery_r2.py",
    "scripts/cls_ppmi_context_recovery_r2.py",
    "tests/test_cls_ppmi_context_recovery_r2.py",
    "scripts/cls_ppmi_context_v1.py",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/PROTOCOL.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/PROTOCOL.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/RECIPE_AUDIT.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/SYNTHETIC_QA.json",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R2/PROTOCOL_R2.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R2/SYNTHETIC_QA_R2.json",
    "experiments/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/CORRECTION_INTENT.json",
    "experiments/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/SCORE_READ_ATTEMPT.json",
    "experiments/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/FAILED_VERIFICATION_R1.json",
)


class RecoveryRunnerError(RuntimeError):
    """Fail-closed error in the one-shot R2 recovery lifecycle."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RecoveryRunnerError(f"R2 source or pinned failure evidence missing: {relative}")
        output[relative] = r1._sha256_file(path)
    return output


def _assert_committed_source_set(intent: Mapping[str, Any]) -> None:
    if intent.get("r2_source_hashes") != _source_hashes():
        raise RecoveryRunnerError("R2 source hashes differ from the committed recovery intent")
    for relative, expected in intent["r2_source_hashes"].items():
        path = ROOT / relative
        r1._assert_committed_exact(path, relative)
        if r1._sha256_file(path) != expected:
            raise RecoveryRunnerError(f"R2 source changed after intent: {relative}")
    for path in (RECOVERY_RECEIPT_PATH, INTENT_PATH):
        r1._assert_committed_exact(path)
    if r1._sha256_file(RECOVERY_RECEIPT_PATH) != intent.get("recovery_receipt_sha256"):
        raise RecoveryRunnerError("Committed root recovery receipt differs from intent")


def _load_frozen_inputs(*, verify_committed: bool = True):
    auth, freeze, train_path = r1._load_authority(verify_committed=verify_committed)
    historical = r1._verify_historical_artifacts(auth, verify_committed=verify_committed)
    predictions = r1._load_predictions(auth)
    selection = r1.core.selection_binding(predictions)
    if selection["selected_rows_count"] != 20_903:
        raise RecoveryRunnerError("R2 selection differs from the authorized 20,903-row primary set")
    return auth, freeze, train_path, historical, predictions, selection


def _read_failure_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not R1_FAILURE_PATH.is_file() or not R1_ATTEMPT_PATH.is_file() or not R1_INTENT_PATH.is_file():
        raise RecoveryRunnerError("R1 failed-attempt, failure, and intent evidence must all be present")
    failure = r1._read_json(R1_FAILURE_PATH)
    attempt = r1._read_json(R1_ATTEMPT_PATH)
    intent = r1._read_json(R1_INTENT_PATH)
    if failure.get("status") != "FAILED_AFTER_ONE_SELECTED_READ_DURING_IN_MEMORY_VERIFICATION" \
            or attempt.get("selected_read_call_count") != 1 \
            or attempt.get("target_values_expected") != 20_903:
        raise RecoveryRunnerError("R1 failure evidence does not record the consumed one-read attempt")
    return failure, attempt, intent


def _production_schema_anchors(scored: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for view_name in ("PRIMARY_UNIQUE", "E67_H12", "E75_W6"):
        metrics = scored["views"][view_name]["metrics"]
        output[view_name] = {
            "champion": float(metrics["champion"]["macro_f1"]),
            "candidate_real": float(metrics["candidate_real"]["macro_f1"]),
            "control_shuffled": float(metrics["control_shuffled"]["macro_f1"]),
            "gain_real_vs_champion": float(
                metrics["gains"]["candidate_real_minus_champion_macro_f1"]),
        }
    return output


def _original_campaign_scientific_gate(scored: Mapping[str, Any]) -> dict[str, Any]:
    r1_checks = dict(scored["gates"]["scientific"]["checks"])
    r1_checks.pop("e75_real_minus_shuffled_nonnegative", None)
    original_e75_control_gain = float(scored["views"]["E75_W6"]["metrics"]["gains"]
                                      ["control_shuffled_minus_champion_macro_f1"])
    r1_checks["e75_control_gain_nonnegative"] = original_e75_control_gain >= 0.0
    return {
        "status": "PASS" if all(r1_checks.values()) else "FAIL",
        "checks": r1_checks,
        "e75_control_gain": original_e75_control_gain,
        "gate_source": "original ROOT_CAMPAIGN / scripts/cls_ppmi_context_v1.py",
    }


def _dual_gate_comparison(scored: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "original_root_campaign_scientific_gate": _original_campaign_scientific_gate(scored),
        "r1_amended_e75_comparison": dual_e75_gate_readback(scored),
        "r1_amended_scientific_gate": {
            "status": scored["gates"]["scientific"]["status"],
            "checks": scored["gates"]["scientific"]["checks"],
            "label": "later root amendment in ROOT_CORRECTED_SCORING_R1.json",
        },
        "exploratory_gate_unchanged": scored["gates"]["exploratory"],
    }


def _run_read_guard(intent: Mapping[str, Any], train_path: Path,
                    selection: Mapping[str, Any], *,
                    committed_check, reader):
    """Run commit validation before calling the sole selected-label reader."""
    committed_check(intent)
    return reader(train_path, selection, expected_rows=20_903)


def _synthetic_read_guard_case() -> bool:
    calls: list[str] = []

    def reject_uncommitted(_intent):
        raise RecoveryRunnerError("synthetic uncommitted source set")

    def reader_would_touch_labels(*_args, **_kwargs):
        calls.append("reader_called")
        return None

    try:
        _run_read_guard({}, Path("synthetic-unused.csv"), {},
                        committed_check=reject_uncommitted,
                        reader=reader_would_touch_labels)
    except RecoveryRunnerError:
        return not calls
    return False


def synthetic_recovery_lifecycle() -> dict[str, Any]:
    """Full synthetic score, exact-schema anchors, independent QA, and signed serialization."""
    predictions, labels_by_id = r1.core.synthetic_prediction_fixture()
    scored = r1.core.score_correction(
        predictions, labels_by_id, prelabel_verification_sha256="synthetic-recovery-r2")
    anchors = _production_schema_anchors(scored)
    independent = verify_in_memory_recovery_r2(predictions, labels_by_id, scored, anchors)
    gate_readback = _dual_gate_comparison(scored)
    synthetic_result = {
        "schema_version": 1,
        "correction_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R2",
        "overall": scored["overall"],
        "corrected_metrics": scored,
        "original_root_campaign_scientific_gate": gate_readback["original_root_campaign_scientific_gate"],
        "r1_amended_scientific_gate": gate_readback["r1_amended_scientific_gate"],
        "dual_e75_gate_comparison": gate_readback["r1_amended_e75_comparison"],
        "independent_verification": independent,
        "test_or_sample_opened": False,
    }
    synthetic_report = _render_report(synthetic_result)
    if "Original control-shuffled minus champion" not in synthetic_report \
            or "R1-amended real candidate minus shuffled control" not in synthetic_report:
        raise RecoveryRunnerError("Synthetic corrected report omitted a separately labeled E75 gate")
    synthetic_result["synthetic_signature"] = r1._canonical_sha256(synthetic_result)
    serialized = r1._json_bytes(synthetic_result)
    reloaded = json.loads(serialized.decode("utf-8"))
    if not r1._check_signature(reloaded, "synthetic_signature"):
        raise RecoveryRunnerError("Synthetic full-result serialization signature did not round-trip")

    mismatched = json.loads(json.dumps(anchors))
    mismatched["E75_W6"]["gain_real_vs_champion"] += 0.01
    try:
        verify_in_memory_recovery_r2(predictions, labels_by_id, scored, mismatched)
    except IndependentVerificationError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    unknown = json.loads(json.dumps(anchors))
    unknown["E75_W6"]["unrecognized_anchor"] = 0.0
    try:
        verify_in_memory_recovery_r2(predictions, labels_by_id, scored, unknown)
    except IndependentVerificationError:
        unknown_rejected = True
    else:
        unknown_rejected = False
    if not mismatch_rejected or not unknown_rejected:
        raise RecoveryRunnerError("Synthetic anchor mismatch/unknown-key rejection failed")
    pre_read_guard_passed = _synthetic_read_guard_case()
    if not pre_read_guard_passed:
        raise RecoveryRunnerError("Synthetic commit guard did not stop before the reader")
    adversarial_predictions, adversarial_labels = r1.core.synthetic_prediction_fixture()
    e75 = adversarial_predictions[75]
    positive_present = np.asarray([
        index for index, record_id in enumerate(e75["score_record_ids"].astype(str))
        if e75["present_mask"][index] and adversarial_labels[record_id] == 1
    ], dtype=np.int64)
    e75["p_candidate_real"][positive_present[:100]] = 0.1
    e75["p_control_shuffled"][positive_present[:220]] = 0.1
    adversarial_scored = r1.core.score_correction(
        adversarial_predictions, adversarial_labels,
        prelabel_verification_sha256="synthetic-r2-opposite-e75-gates")
    adversarial_gates = dual_e75_gate_readback(adversarial_scored)
    if adversarial_gates["original_root_campaign"]["nonnegative"] == \
            adversarial_gates["r1_amended_comparison"]["nonnegative"]:
        raise RecoveryRunnerError("Synthetic original/amended E75 gate signs must differ")

    views = scored["views"]
    qa = {
        "schema_version": 1,
        "correction_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R2",
        "status": "PASS",
        "official_source_opened": False,
        "query_targets_read": False,
        "test_or_sample_opened": False,
        "complete_synthetic_lifecycle": {
            "score_views": len(views),
            "view_names": sorted(views),
            "all_view_guards_recomputed": independent["all_supported_subgroups_recomputed_across_all_views"],
            "production_anchor_schema_views": sorted(anchors),
            "production_anchor_schema_keys": sorted(next(iter(anchors.values()))),
            "gain_anchor_derived_metric": "candidate_real_minus_champion_macro_f1",
            "independent_verifier": independent["status"],
            "serialization_roundtrip_signature": "PASS",
            "report_generation": "PASS",
            "mismatched_anchor_rejected": mismatch_rejected,
            "unknown_anchor_rejected": unknown_rejected,
        },
        "dual_e75_gate_fixtures": {
            "baseline": dual_e75_gate_readback(scored),
            "opposite_sign_adversarial": adversarial_gates,
            "signs_differ": True,
        },
        "synthetic_result_sha256": hashlib.sha256(serialized).hexdigest(),
        "pre_read_guard_case": "PASS" if pre_read_guard_passed else "FAIL",
    }
    if len(views) != 6 or len(independent["macro_f1_root_anchors"]) != 3:
        raise RecoveryRunnerError("Synthetic six-view / three-view anchor coverage is incomplete")
    return qa


def command_synthetic() -> None:
    if SYNTHETIC_QA_PATH.exists():
        raise RecoveryRunnerError("R2 synthetic QA is write-once and already exists")
    qa = synthetic_recovery_lifecycle()
    r1._atomic_json(SYNTHETIC_QA_PATH, qa)
    print(json.dumps({"status": "PASS", "path": str(SYNTHETIC_QA_PATH),
                      "sha256": r1._sha256_file(SYNTHETIC_QA_PATH),
                      "views": qa["complete_synthetic_lifecycle"]["score_views"],
                      "pre_read_guard": qa["pre_read_guard_case"]}, sort_keys=True))


def command_prepare_recovery_receipt() -> None:
    if RECOVERY_RECEIPT_PATH.exists():
        raise RecoveryRunnerError("R2 root recovery receipt is write-once")
    auth, freeze, _, historical, _, selection = _load_frozen_inputs()
    r1_attempt = r1._read_json(R1_ATTEMPT_PATH)
    r1_intent = r1._read_json(R1_INTENT_PATH)
    if r1_attempt.get("selected_read_call_count") != 1 \
            or r1_attempt.get("selected_rows") != selection["selected_rows_count"] \
            or r1_attempt.get("selected_rows_sha256") != selection["selected_rows_sha256"] \
            or r1_attempt.get("selected_ids_sha256") != selection["selected_ids_sha256"] \
            or r1_attempt.get("intent_signature") != r1_intent.get("intent_signature") \
            or r1_intent.get("selected_primary_rows", {}).get("rows_sha256") != selection["selected_rows_sha256"] \
            or r1_intent.get("selected_primary_rows", {}).get("record_ids_sha256") != selection["selected_ids_sha256"]:
        raise RecoveryRunnerError("R1 attempt/intent does not bind the same selected row and ID set")
    failure = {
        "schema_version": 1,
        "correction_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R1",
        "status": "FAILED_AFTER_ONE_SELECTED_READ_DURING_IN_MEMORY_VERIFICATION",
        "r1_commit": "38cae991b917413b475bc56931efe5a8412f1aef",
        "r1_intent_signature": r1._read_json(R1_INTENT_PATH)["intent_signature"],
        "selected_read_call_count": 1,
        "selected_target_values_materialized": 20_903,
        "additional_verifier_target_reads": 0,
        "error_type": "KeyError",
        "error_message": "'gain_real_vs_champion'",
        "failure_location": "scripts/verify_cls_ppmi_context_correction_r1.py:351",
        "cause": "The verifier indexed a derived gain anchor as a model metric instead of mapping it to candidate_real_minus_champion_macro_f1.",
        "result_report_verification_and_completion_receipt_written": False,
        "selected_target_values_persisted": False,
        "old_frozen_artifacts_changed": False,
        "retry_in_r1_namespace_allowed": False,
    }
    if R1_FAILURE_PATH.exists():
        if r1._read_json(R1_FAILURE_PATH) != failure:
            raise RecoveryRunnerError("Existing R1 failure evidence differs from the verified failure")
    else:
        r1._atomic_json(R1_FAILURE_PATH, failure)
    failed_attempt = r1._read_json(R1_ATTEMPT_PATH)
    r1_intent = r1._read_json(R1_INTENT_PATH)
    body = {
        "schema_version": 1,
        "recovery_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R2",
        "status": "ROOT_APPROVED_ONE_SELECTED_READ_ZERO_FITS",
        "root_decision": {
            "thread_id": ROOT_DECISION_THREAD_ID,
            "message": ROOT_DECISION_MESSAGE,
            "message_sha256": _sha256_text(ROOT_DECISION_MESSAGE),
        },
        "root_campaign_historical_verdict": auth["original_verdict_status"],
        "source_freeze_signature": freeze["freeze_signature"],
        "source_freeze_file_sha256": r1._sha256_file(r1.FREEZE_PATH),
        "official_train_sha256": freeze["official_source"]["sha256"],
        "prediction_hashes": auth["prediction_hashes"],
        "original_historical_artifact_hashes": historical,
        "r1_failed_attempt": {
            "r1_intent_sha256": r1._sha256_file(R1_INTENT_PATH),
            "r1_intent_signature": r1_intent["intent_signature"],
            "r1_attempt_sha256": r1._sha256_file(R1_ATTEMPT_PATH),
            "r1_attempt_status": failed_attempt["status"],
            "r1_failure_record_sha256": r1._sha256_file(R1_FAILURE_PATH),
            "selected_rows": selection["selected_rows_count"],
            "selected_rows_sha256": selection["selected_rows_sha256"],
            "selected_ids_sha256": selection["selected_ids_sha256"],
        },
        "r2_authorized_read": {
            "selected_read_calls": 1,
            "selected_rows": 20_903,
            "same_r1_rows_and_ids": True,
            "additional_fits": 0,
            "automatic_retries": False,
            "target_values_stored": False,
            "test_or_sample_opened": False,
            "kaggle_opened": False,
        },
        "expected_cumulative_read_accounting": {
            "target_column_read_calls": 3,
            "target_values_materialized": 92_414,
            "unique_source_rows": 50_608,
            "r1_selected_values_reloaded": 20_903,
            "r2_selected_values_reloaded": 20_903,
            "additional_verifier_reads": 0,
        },
        "gate_semantics": {
            "original_root_campaign_e75": "control_shuffled_minus_champion_macro_f1 >= 0",
            "r1_amended_e75_reported_separately": "candidate_real_minus_control_shuffled_macro_f1 >= 0",
            "exploratory_eligibility": "unchanged from original root campaign",
        },
        "corrected_source_hashes": _source_hashes(),
        "synthetic_qa_sha256": r1._sha256_file(SYNTHETIC_QA_PATH),
        "created_at_utc": r1._utc_now(),
    }
    body["recovery_receipt_signature"] = r1._canonical_sha256(body)
    r1._atomic_json(RECOVERY_RECEIPT_PATH, body)
    print(json.dumps({"status": body["status"], "path": str(RECOVERY_RECEIPT_PATH),
                      "receipt_sha256": r1._sha256_file(RECOVERY_RECEIPT_PATH),
                      "r1_failed_read_bound": body["r1_failed_attempt"]["selected_rows"],
                      "r2_reads_authorized": body["r2_authorized_read"]["selected_read_calls"],
                      "source_paths": len(body["corrected_source_hashes"])}, sort_keys=True))


def command_prepare_intent() -> None:
    if INTENT_PATH.exists() or ATTEMPT_PATH.exists() or any(
            p.exists() for p in (RESULTS_PATH, REPORT_PATH, VERIFICATION_PATH, RECEIPT_PATH)):
        raise RecoveryRunnerError("R2 intent/output namespace must be empty before preparation")
    if not RECOVERY_RECEIPT_PATH.is_file() or not r1._check_signature(
            r1._read_json(RECOVERY_RECEIPT_PATH), "recovery_receipt_signature"):
        raise RecoveryRunnerError("Committed root recovery receipt is required before intent")
    auth, freeze, train_path, historical, predictions, selection = _load_frozen_inputs()
    failure, attempt, r1_intent = _read_failure_evidence()
    receipt = r1._read_json(RECOVERY_RECEIPT_PATH)
    if receipt["corrected_source_hashes"] != _source_hashes():
        raise RecoveryRunnerError("Recovery receipt source hashes differ from current R2 sources")
    if receipt["r1_failed_attempt"]["r1_intent_signature"] != r1_intent["intent_signature"] \
            or receipt["r1_failed_attempt"]["r1_attempt_sha256"] != r1._sha256_file(R1_ATTEMPT_PATH):
        raise RecoveryRunnerError("Recovery receipt does not bind the consumed R1 attempt")
    body = {
        "schema_version": 1,
        "recovery_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R2",
        "status": "AUTHORIZED_NOT_READ",
        "recovery_receipt_sha256": r1._sha256_file(RECOVERY_RECEIPT_PATH),
        "recovery_receipt_signature": receipt["recovery_receipt_signature"],
        "authorization_sha256": r1._sha256_file(r1.AUTH_PATH),
        "source_freeze_signature": freeze["freeze_signature"],
        "official_train_sha256": freeze["official_source"]["sha256"],
        "prediction_hashes": auth["prediction_hashes"],
        "historical_artifact_hashes": historical,
        "r1_failed_record_sha256": r1._sha256_file(R1_FAILURE_PATH),
        "r1_attempt_sha256": r1._sha256_file(R1_ATTEMPT_PATH),
        "selected_primary_rows": {
            "count": selection["selected_rows_count"],
            "rows_sha256": selection["selected_rows_sha256"],
            "record_ids_sha256": selection["selected_ids_sha256"],
            "origins": selection["origins"],
            "row_order": "source Train row order ascending",
        },
        "r1_read_accounting_before_r2": {
            "original_full_column_read_calls": auth["original_actual_target_column_read_count"],
            "original_values_materialized": auth["original_actual_target_vector_rows"],
            "r1_failed_selected_read_calls": attempt["selected_read_call_count"],
            "r1_failed_selected_values_materialized": failure["selected_target_values_materialized"],
            "additional_verifier_reads": failure["additional_verifier_target_reads"],
        },
        "r2_target_values_materialized_before_read": 0,
        "r2_source_hashes": _source_hashes(),
        "synthetic_qa_sha256": r1._sha256_file(SYNTHETIC_QA_PATH),
        "created_at_utc": r1._utc_now(),
    }
    body["intent_signature"] = r1._canonical_sha256(body)
    r1._atomic_json(INTENT_PATH, body)
    print(json.dumps({"status": body["status"], "path": str(INTENT_PATH),
                      "intent_sha256": r1._sha256_file(INTENT_PATH),
                      "selected_rows": selection["selected_rows_count"],
                      "selected_ids_sha256": selection["selected_ids_sha256"],
                      "target_values_materialized_before_read": 0}, sort_keys=True))


def _read_after_commit_guard(intent: Mapping[str, Any], train_path: Path,
                             selection: Mapping[str, Any]):
    """The sole target-reader entry; fail before invoking pandas if source commit drifts."""
    return _run_read_guard(intent, train_path, selection,
                           committed_check=_assert_committed_source_set,
                           reader=r1._read_selected_labels_once)


def _assert_one_shot_namespace_clean() -> None:
    if ATTEMPT_PATH.exists() or any(path.exists() for path in
                                    (RESULTS_PATH, REPORT_PATH, VERIFICATION_PATH, RECEIPT_PATH)):
        raise RecoveryRunnerError("R2 one-read attempt already exists; automatic retry is forbidden")


def _build_read_accounting(auth: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "original_evaluation_target_column_reads": int(auth["original_actual_target_column_read_count"]),
        "original_evaluation_target_values_materialized": int(auth["original_actual_target_vector_rows"]),
        "original_unique_primary_rows_scored": 20_903,
        "original_nonprimary_target_values_also_materialized":
            int(auth["original_actual_target_vector_rows"]) - 20_903,
        "r1_failed_selected_target_column_reads": 1,
        "r1_failed_selected_target_values_materialized": 20_903,
        "r1_additional_verifier_target_reads": 0,
        "r2_selected_target_column_reads": 1,
        "r2_selected_target_values_materialized": 20_903,
        "cumulative_target_column_reads": 3,
        "cumulative_target_values_materialized": 92_414,
        "unique_source_rows_across_all_passes": 50_608,
        "selected_rows_reloaded_by_r1_and_r2": 41_806,
        "r2_additional_verifier_target_reads": 0,
        "target_values_stored_in_result": False,
    }


def _render_report(result: Mapping[str, Any]) -> str:
    metrics = result["corrected_metrics"]
    lines = [
        "# CLS_PPMI_CONTEXT_V1 corrected scoring — CORRECTION_R2", "",
        f"Corrected original-campaign scientific gate: **{result['original_root_campaign_scientific_gate']['status']}**.",
        f"Exploratory eligibility (unchanged): **{metrics['gates']['exploratory']['status']}**.", "",
        "The original R1 result/report remain preserved as history. R1 consumed its one selected read but failed during in-memory root-anchor verification; R2 is the separately authorized recovery pass.", "",
        "## Corrected MacroF1 views", "",
        "| View | Rows | Champion | Real candidate | Shuffled control | Real − champion | Control − champion | Real − control |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in metrics["views"].items():
        m = view["metrics"]
        g = m["gains"]
        lines.append(
            f"| {name} | {view['rows']} | {m['champion']['macro_f1']:.9f} | "
            f"{m['candidate_real']['macro_f1']:.9f} | {m['control_shuffled']['macro_f1']:.9f} | "
            f"{g['candidate_real_minus_champion_macro_f1']:+.9f} | "
            f"{g['control_shuffled_minus_champion_macro_f1']:+.9f} | "
            f"{g['candidate_real_minus_control_shuffled_macro_f1']:+.9f} |"
        )
    dual = result["dual_e75_gate_comparison"]
    lines += [
        "", "## E75 gate semantics", "",
        "The original ROOT_CAMPAIGN E75 gate is control-shuffled minus champion >= 0. R1 introduced a later amended comparison, real candidate minus shuffled control >= 0. Both are shown separately; R2 preserves the original gate as the scientific decision contract.",
        f"- Original control-shuffled minus champion: {dual['original_root_campaign']['gain']:+.9f} — **{'PASS' if dual['original_root_campaign']['nonnegative'] else 'FAIL'}**.",
        f"- R1-amended real candidate minus shuffled control: {dual['r1_amended_comparison']['gain']:+.9f} — **{'PASS' if dual['r1_amended_comparison']['nonnegative'] else 'FAIL'}**.",
        f"- Original corrected scientific gate: **{result['original_root_campaign_scientific_gate']['status']}**.",
        f"- R1-amended scientific gate: **{result['r1_amended_scientific_gate']['status']}**.", "",
        "## All-view guards", "",
        f"- Worst supported subgroup MacroF1 harm: {metrics['gates']['scientific']['worst_supported_macro_f1_harm_all_views']:.9f}.",
        f"- Worst supported subgroup positive-recall harm: {metrics['gates']['scientific']['worst_supported_positive_recall_harm_all_views']:.9f}.",
        f"- Supported groups by view: `{json.dumps(metrics['gates']['scientific']['supported_groups_by_view'], sort_keys=True)}`.", "",
        "## Read accounting", "",
        "- Original evaluation: 50,608 target values in one read.",
        "- Failed R1 correction: 20,903 selected values in one read; no result was persisted.",
        "- R2 recovery: 20,903 selected values in one read; independent verification reused this vector.",
        "- Cumulative: 3 target-column read calls and 92,414 materialized target values; 50,608 unique Train rows; 0 additional verifier reads.",
        "- No fit, prediction, policy, threshold, or support change. Test/Kaggle were not opened.", "",
        "## Decision boundary", "",
        "This report corrects scoring and gate accounting only. The historical original output remains invalid for the intended MacroF1/windows/subgroup contract, and the root retains family disposition and next-lane ownership.", "",
    ]
    return "\n".join(lines)


def command_score() -> None:
    _assert_one_shot_namespace_clean()
    if not INTENT_PATH.is_file() or not r1._check_signature(
            r1._read_json(INTENT_PATH), "intent_signature"):
        raise RecoveryRunnerError("Committed R2 recovery intent/signature required")
    intent = r1._read_json(INTENT_PATH)
    recovery_receipt = r1._read_json(RECOVERY_RECEIPT_PATH)
    if not r1._check_signature(recovery_receipt, "recovery_receipt_signature"):
        raise RecoveryRunnerError("R2 root recovery receipt signature mismatch")
    _assert_committed_source_set(intent)
    auth, freeze, train_path, historical, predictions, selection = _load_frozen_inputs()
    failure, r1_attempt, _ = _read_failure_evidence()
    if historical != intent["historical_artifact_hashes"]:
        raise RecoveryRunnerError("Frozen historical artifacts differ from R2 intent")
    if selection["selected_rows_sha256"] != intent["selected_primary_rows"]["rows_sha256"] \
            or selection["selected_ids_sha256"] != intent["selected_primary_rows"]["record_ids_sha256"]:
        raise RecoveryRunnerError("R2 selected row/ID binding differs from committed intent")
    if r1_attempt["selected_read_call_count"] != 1 \
            or failure["selected_target_values_materialized"] != 20_903:
        raise RecoveryRunnerError("Consumed R1 read accounting is not pinned")
    if r1._sha256_file(train_path) != freeze["official_source"]["sha256"]:
        raise RecoveryRunnerError("Frozen official Train source hash changed before R2 selected read")
    attempt = {
        "schema_version": 1, "recovery_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R2",
        "status": "ONE_R2_SELECTED_READ_ATTEMPT_STARTED",
        "intent_signature": intent["intent_signature"],
        "recovery_receipt_signature": recovery_receipt["recovery_receipt_signature"],
        "source_sha256": freeze["official_source"]["sha256"],
        "selected_rows": selection["selected_rows_count"],
        "selected_rows_sha256": selection["selected_rows_sha256"],
        "selected_ids_sha256": selection["selected_ids_sha256"],
        "target_values_expected": 20_903, "selected_read_call_count": 1,
        "created_at_utc": r1._utc_now(),
    }
    r1._atomic_json(ATTEMPT_PATH, attempt)
    # Any failure after this marker consumes the sole R2 authorization; no retry path exists.
    observed_ids, selected_labels = _read_after_commit_guard(intent, train_path, selection)
    labels_by_id = r1.core.labels_to_map(observed_ids, selected_labels)
    target_sha = hashlib.sha256(np.ascontiguousarray(selected_labels.astype(np.uint8)).tobytes()).hexdigest()
    metrics = r1.core.score_correction(
        predictions, labels_by_id,
        prelabel_verification_sha256=historical["prelabel_verification"])
    anchors = auth["no_label_root_checks"]["confusion_matrix_only_macro_reconstruction"]
    independent = verify_in_memory_recovery_r2(predictions, labels_by_id, metrics, anchors)
    fit_summary = r1._read_json(r1.FIT_SUMMARY_PATH)
    dual_gate = _dual_gate_comparison(metrics)
    original_gate = dual_gate["original_root_campaign_scientific_gate"]
    original_overall = ("SCIENTIFIC_PASS" if original_gate["status"] == "PASS"
                        else "EXPLORATORY_ONLY_ELIGIBLE"
                        if metrics["gates"]["exploratory"]["status"] == "ELIGIBLE"
                        else "NO_ADVANCE")
    accounting = _build_read_accounting(auth)
    body = {
        "schema_version": 1,
        "recovery_id": "CLS_PPMI_CONTEXT_V1_CORRECTION_R2",
        "authorization_receipt_sha256": r1._sha256_file(RECOVERY_RECEIPT_PATH),
        "authorization_receipt_signature": recovery_receipt["recovery_receipt_signature"],
        "historical_original_output_disposition": auth["original_verdict_status"],
        "source_freeze_signature": freeze["freeze_signature"],
        "prediction_hashes": auth["prediction_hashes"],
        "r1_failed_attempt_sha256": r1._sha256_file(R1_ATTEMPT_PATH),
        "r1_failure_record_sha256": r1._sha256_file(R1_FAILURE_PATH),
        "query_target_values_read_in_r2": 20_903,
        "query_target_selected_rows_sha256": selection["selected_rows_sha256"],
        "query_target_selected_ids_sha256": selection["selected_ids_sha256"],
        "query_target_vector_sha256": target_sha,
        "read_accounting": accounting,
        "fit_accounting": {"additional_fits": 0, "original_representation_fits": 4,
                           "original_predictor_fits": 4, "original_fit_retries": 0},
        "corrected_metrics": metrics,
        "original_root_campaign_scientific_gate": original_gate,
        "r1_amended_scientific_gate": dual_gate["r1_amended_scientific_gate"],
        "dual_e75_gate_comparison": dual_gate["r1_amended_e75_comparison"],
        "exploratory_gate_unchanged": metrics["gates"]["exploratory"],
        "independent_verification": independent,
        "original_campaign_corrected_overall": original_overall,
        "test_or_sample_opened": False,
        "kaggle_opened": False,
        "family_or_next_lane_decision": "ROOT_REVIEW_REQUIRED",
        "created_at_utc": r1._utc_now(),
    }
    body["correction_signature"] = r1._canonical_sha256(body)
    report = _render_report(body)
    r1._atomic_json(RESULTS_PATH, body)
    r1._atomic_write(REPORT_PATH, report.encode("utf-8"))
    verification = {
        "schema_version": 1, "recovery_id": body["recovery_id"],
        "status": independent["status"], "independent_verification": independent,
        "selected_target_rows_in_memory": len(labels_by_id),
        "selected_target_vector_sha256": target_sha,
        "target_read_calls_in_verifier": 0,
        "created_at_utc": r1._utc_now(),
    }
    verification["verification_signature"] = r1._canonical_sha256(verification)
    r1._atomic_json(VERIFICATION_PATH, verification)
    receipt = {
        "schema_version": 1, "recovery_id": body["recovery_id"],
        "status": "PASS", "intent_signature": intent["intent_signature"],
        "recovery_receipt_signature": recovery_receipt["recovery_receipt_signature"],
        "result_sha256": r1._sha256_file(RESULTS_PATH),
        "report_sha256": r1._sha256_file(REPORT_PATH),
        "verification_sha256": r1._sha256_file(VERIFICATION_PATH),
        "attempt_sha256": r1._sha256_file(ATTEMPT_PATH),
        "read_accounting": accounting,
        "independent_verifier_status": independent["status"],
        "r2_selected_target_read_once": True,
        "additional_verifier_reads": 0,
        "test_or_sample_opened": False,
        "kaggle_opened": False,
        "created_at_utc": r1._utc_now(),
    }
    receipt["receipt_signature"] = r1._canonical_sha256(receipt)
    r1._atomic_json(RECEIPT_PATH, receipt)
    print(json.dumps({
        "status": receipt["status"], "original_campaign_overall": original_overall,
        "r1_amended_scientific_gate": dual_gate["r1_amended_scientific_gate"]["status"],
        "exploratory_gate": metrics["gates"]["exploratory"]["status"],
        "result_path": str(RESULTS_PATH),
        "r2_target_values_read": 20_903,
        "cumulative_target_values_materialized": 92_414,
        "additional_verifier_reads": 0,
    }, sort_keys=True))


def command_verify() -> None:
    auth, _, _, historical, _, _ = _load_frozen_inputs()
    for path in (RECOVERY_RECEIPT_PATH, INTENT_PATH, ATTEMPT_PATH, RESULTS_PATH,
                 REPORT_PATH, VERIFICATION_PATH, RECEIPT_PATH, R1_ATTEMPT_PATH,
                 R1_FAILURE_PATH):
        r1._assert_committed_exact(path)
    intent = r1._read_json(INTENT_PATH)
    root_receipt = r1._read_json(RECOVERY_RECEIPT_PATH)
    attempt = r1._read_json(ATTEMPT_PATH)
    results = r1._read_json(RESULTS_PATH)
    verification = r1._read_json(VERIFICATION_PATH)
    receipt = r1._read_json(RECEIPT_PATH)
    _assert_committed_source_set(intent)
    for item, signature_key in ((intent, "intent_signature"),
                                (root_receipt, "recovery_receipt_signature"),
                                (results, "correction_signature"),
                                (verification, "verification_signature"),
                                (receipt, "receipt_signature")):
        if not r1._check_signature(item, signature_key):
            raise RecoveryRunnerError(f"Committed R2 {signature_key} is invalid")
    accounting = _build_read_accounting(auth)
    if results.get("read_accounting") != accounting or receipt.get("read_accounting") != accounting:
        raise RecoveryRunnerError("Cumulative 3-read accounting differs from root authorization")
    if attempt.get("selected_read_call_count") != 1 \
            or attempt.get("target_values_expected") != 20_903 \
            or verification.get("target_read_calls_in_verifier") != 0 \
            or receipt.get("additional_verifier_reads") != 0:
        raise RecoveryRunnerError("R2 one-read/no-verifier-read receipts do not match authorization")
    if historical != intent.get("historical_artifact_hashes"):
        raise RecoveryRunnerError("Original frozen result/report/receipt hashes changed")
    if results.get("historical_original_output_disposition") != auth["original_verdict_status"] \
            or results.get("family_or_next_lane_decision") != "ROOT_REVIEW_REQUIRED":
        raise RecoveryRunnerError("R2 result changed historical verdict or root decision boundary")
    if results.get("original_root_campaign_scientific_gate", {}).get("status") not in {"PASS", "FAIL"}:
        raise RecoveryRunnerError("Original root campaign E75 gate was not separately reported")
    if results.get("dual_e75_gate_comparison", {}).get("original_root_campaign", {}).get(
            "comparison") != "control_shuffled_minus_champion_macro_f1":
        raise RecoveryRunnerError("Original E75 control-vs-champion comparison is missing")
    if results.get("dual_e75_gate_comparison", {}).get("r1_amended_comparison", {}).get(
            "comparison") != "candidate_real_minus_control_shuffled_macro_f1":
        raise RecoveryRunnerError("R1-amended E75 comparison is missing")
    for path, key in ((RESULTS_PATH, "result_sha256"), (REPORT_PATH, "report_sha256"),
                      (VERIFICATION_PATH, "verification_sha256"), (ATTEMPT_PATH, "attempt_sha256")):
        if r1._sha256_file(path) != receipt[key]:
            raise RecoveryRunnerError(f"R2 committed artifact hash mismatch: {path.name}")
    print(json.dumps({"status": "PASS", "phase": "committed_structural_no_label_reread",
                      "original_campaign_overall": results["original_campaign_corrected_overall"],
                      "results_sha256": r1._sha256_file(RESULTS_PATH),
                      "cumulative_target_reads": accounting["cumulative_target_column_reads"],
                      "cumulative_target_values": accounting["cumulative_target_values_materialized"],
                      "additional_verifier_reads": 0}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic", help="Run full synthetic scoring/verification/serialization QA")
    sub.add_parser("prepare-recovery", help="Bind R1 failure and root R2 authorization without labels")
    sub.add_parser("prepare-intent", help="Bind exact R2 selection without target labels")
    sub.add_parser("score", help="Consume exactly one new authorized R2 selected read")
    sub.add_parser("verify", help="Verify committed R2 results without rereading labels")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            command_synthetic()
        elif args.command == "prepare-recovery":
            command_prepare_recovery_receipt()
        elif args.command == "prepare-intent":
            command_prepare_intent()
        elif args.command == "score":
            command_score()
        elif args.command == "verify":
            command_verify()
        return 0
    except (RecoveryRunnerError, r1.CorrectionRunnerError, r1.core.CorrectionError,
            IndependentVerificationError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
