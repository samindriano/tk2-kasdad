"""Additive, zero-fit correction for CLS_PPMI_CONTEXT_V1 scoring/accounting."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = ROOT / "src" / "models"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import cls_ppmi_context_correction_r1 as core  # noqa: E402
from verify_cls_ppmi_context_correction_r1 import (  # noqa: E402
    IndependentVerificationError,
    synthetic_gate_verification,
    verify_in_memory,
)

BASE_REPORT_DIR = ROOT / "reports/optimization_v3/CLS_PPMI_CONTEXT_V1"
BASE_EXPERIMENT_DIR = ROOT / "experiments/CLS_PPMI_CONTEXT_V1"
CORRECTION_REPORT_DIR = BASE_REPORT_DIR / "CORRECTION_R1"
CORRECTION_EXPERIMENT_DIR = BASE_EXPERIMENT_DIR / "CORRECTION_R1"
AUTH_PATH = BASE_REPORT_DIR / "ROOT_CORRECTED_SCORING_R1.json"
FREEZE_PATH = BASE_REPORT_DIR / "SOURCE_FREEZE.json"
PRELABEL_PATH = BASE_EXPERIMENT_DIR / "PRELABEL_READY.json"
PRELABEL_VERIFICATION_PATH = BASE_EXPERIMENT_DIR / "PRELABEL_VERIFICATION.json"
FIT_SUMMARY_PATH = BASE_EXPERIMENT_DIR / "FIT_SUMMARY.json"
TRAIN_RELATIVE = Path("data/raw/classification/Train.csv")

SYNTHETIC_QA_PATH = CORRECTION_REPORT_DIR / "SYNTHETIC_QA.json"
RECIPE_AUDIT_PATH = CORRECTION_REPORT_DIR / "RECIPE_AUDIT.md"
PROTOCOL_PATH = CORRECTION_REPORT_DIR / "PROTOCOL.md"
INTENT_PATH = CORRECTION_EXPERIMENT_DIR / "CORRECTION_INTENT.json"
ATTEMPT_PATH = CORRECTION_EXPERIMENT_DIR / "SCORE_READ_ATTEMPT.json"
RESULTS_PATH = CORRECTION_EXPERIMENT_DIR / "CORRECTED_RESULTS.json"
REPORT_PATH = CORRECTION_EXPERIMENT_DIR / "CORRECTED_REPORT.md"
MEMORY_VERIFICATION_PATH = CORRECTION_EXPERIMENT_DIR / "IN_MEMORY_VERIFICATION.json"
RECEIPT_PATH = CORRECTION_EXPERIMENT_DIR / "CORRECTION_RECEIPT.json"

SOURCE_PATHS = (
    "src/models/cls_ppmi_context_correction_r1.py",
    "scripts/cls_ppmi_context_correction_r1.py",
    "scripts/verify_cls_ppmi_context_correction_r1.py",
    "tests/test_cls_ppmi_context_correction_r1.py",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/PROTOCOL.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/RECIPE_AUDIT.md",
    "reports/optimization_v3/CLS_PPMI_CONTEXT_V1/CORRECTION_R1/SYNTHETIC_QA.json",
)


class CorrectionRunnerError(RuntimeError):
    """Fail-closed correction runner error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      indent=2).encode("utf-8") + b"\n"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments],
                                   text=True, encoding="utf-8").strip()


def _assert_committed_exact(path: Path, relative: str | None = None) -> None:
    relative = relative or path.relative_to(ROOT).as_posix()
    try:
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])
    except subprocess.CalledProcessError as exc:
        raise CorrectionRunnerError(f"Required file is not committed: {relative}") from exc
    if not path.is_file() or path.read_bytes() != committed:
        raise CorrectionRunnerError(f"Committed and on-disk bytes differ: {relative}")


def _atomic_write(path: Path, payload: bytes, *, write_once: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_once and path.exists():
        raise CorrectionRunnerError(f"Write-once correction artifact already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if write_once and path.exists():
            raise CorrectionRunnerError(f"Correction artifact appeared during write: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any, *, write_once: bool = True) -> None:
    _atomic_write(path, _json_bytes(value), write_once=write_once)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_authority(*, verify_committed: bool) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if verify_committed:
        _assert_committed_exact(AUTH_PATH)
        _assert_committed_exact(FREEZE_PATH)
    auth = _read_json(AUTH_PATH)
    if auth.get("status") != "APPROVE_ONE_CORRECTED_METRIC_VIEW_GATE_RECONSTRUCTION":
        raise CorrectionRunnerError("Root correction authorization is not an approval")
    if int(auth.get("additional_predictor_or_representation_fits_allowed", -1)) != 0 \
            or int(auth.get("additional_evaluation_label_vector_reads_allowed", -1)) != 1:
        raise CorrectionRunnerError("Root authorization does not bind a zero-fit, one-read correction")
    freeze = _read_json(FREEZE_PATH)
    body = dict(freeze)
    signature = body.pop("freeze_signature", None)
    if signature != _canonical_sha256(body) or signature != auth.get("freeze_signature"):
        raise CorrectionRunnerError("Frozen source receipt differs from correction authorization")
    train_path = Path(freeze["source_root"]) / TRAIN_RELATIVE
    if freeze["official_source"]["sha256"] != "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c":
        raise CorrectionRunnerError("Frozen Train source hash differs from the authorized input")
    return auth, freeze, train_path


def _verify_historical_artifacts(auth: Mapping[str, Any], *, verify_committed: bool) -> dict[str, str]:
    pinned = {
        "original_results": (BASE_EXPERIMENT_DIR / "RESULTS.json", auth["original_results_sha256"]),
        "original_report": (BASE_EXPERIMENT_DIR / "REPORT.md", auth["original_report_sha256"]),
        "original_postlabel_verification": (BASE_EXPERIMENT_DIR / "POSTLABEL_VERIFICATION.json",
                                            auth["original_postlabel_verification_sha256"]),
        "prediction_E67": (BASE_EXPERIMENT_DIR / "predictions_E67.npz", auth["prediction_hashes"]["67"]),
        "prediction_E75": (BASE_EXPERIMENT_DIR / "predictions_E75.npz", auth["prediction_hashes"]["75"]),
    }
    digests: dict[str, str] = {}
    for name, (path, expected) in pinned.items():
        actual = _sha256_file(path)
        if actual != expected:
            raise CorrectionRunnerError(f"Authorized immutable input hash changed: {name}")
        if verify_committed:
            _assert_committed_exact(path)
        digests[name] = actual
    for path in (PRELABEL_PATH, PRELABEL_VERIFICATION_PATH, FIT_SUMMARY_PATH):
        if verify_committed:
            _assert_committed_exact(path)
    prelabel = _read_json(PRELABEL_PATH)
    prelabel_verification = _read_json(PRELABEL_VERIFICATION_PATH)
    if prelabel.get("freeze_signature") != auth["freeze_signature"] \
            or prelabel.get("query_targets_read") is not False \
            or prelabel_verification.get("status") != "PASS" \
            or prelabel_verification.get("query_targets_read") is not False:
        raise CorrectionRunnerError("Committed prelabel receipt does not satisfy the no-label boundary")
    digests["prelabel"] = _sha256_file(PRELABEL_PATH)
    digests["prelabel_verification"] = _sha256_file(PRELABEL_VERIFICATION_PATH)
    return digests


def _load_predictions(auth: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    predictions: dict[int, dict[str, np.ndarray]] = {}
    for origin in core.ORIGINS:
        path = BASE_EXPERIMENT_DIR / f"predictions_E{origin}.npz"
        expected = auth["prediction_hashes"][str(origin)]
        if _sha256_file(path) != expected:
            raise CorrectionRunnerError(f"Frozen prediction SHA mismatch for E{origin}")
        with np.load(path, allow_pickle=False) as archive:
            if any("label" in name.lower() or name == "artifact_signal" for name in archive.files):
                raise CorrectionRunnerError(f"E{origin} prediction archive includes label data")
            predictions[origin] = {name: archive[name].copy() for name in archive.files}
    core.validate_predictions(predictions)
    return predictions


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise CorrectionRunnerError(f"Correction source is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _check_signature(body_with_signature: Mapping[str, Any], key: str) -> bool:
    body = dict(body_with_signature)
    signature = body.pop(key, None)
    return signature == _canonical_sha256(body)


def command_synthetic() -> None:
    if SYNTHETIC_QA_PATH.exists():
        raise CorrectionRunnerError("Synthetic QA is write-once and already exists")
    core_qa = core.synthetic_qa()
    gate_fixture = synthetic_gate_verification()
    predictions, labels = core.synthetic_prediction_fixture()
    scored = core.score_correction(predictions, labels,
                                   prelabel_verification_sha256="synthetic-prelabel-receipt")
    independent = verify_in_memory(predictions, labels, scored, anchors={})
    if len(scored["views"]) != 6 or set(scored["subgroups_by_view"]) != set(core.VIEW_NAMES):
        raise CorrectionRunnerError("Synthetic six-view/all-view subgroup lifecycle failed")
    qa = {
        "schema_version": 1, "correction_id": core.CORRECTION_ID,
        "status": "PASS", "official_source_opened": False,
        "query_targets_read": False, "test_or_sample_opened": False,
        "cases": {
            **core_qa["cases"],
            "exact_six_view_counts_ids_and_sources": "PASS",
            "all_six_view_subgroups_and_metric_readback": "PASS",
            "independent_sklearn_macro_f1_and_gate_recompute": "PASS",
            "e75_real_minus_shuffle_opposite_sign_fixture": "PASS",
        },
        "window_counts": {name: view["rows"] for name, view in scored["views"].items()},
        "independent_verifier": independent,
        "e75_gate_direction_fixture": gate_fixture,
    }
    qa["case_count"] = len(qa["cases"])
    _atomic_json(SYNTHETIC_QA_PATH, qa)
    print(json.dumps({"status": "PASS", "path": str(SYNTHETIC_QA_PATH),
                      "sha256": _sha256_file(SYNTHETIC_QA_PATH),
                      "case_count": qa["case_count"]}, sort_keys=True))


def command_prepare_intent() -> None:
    if INTENT_PATH.exists() or ATTEMPT_PATH.exists() or RESULTS_PATH.exists() \
            or REPORT_PATH.exists() or MEMORY_VERIFICATION_PATH.exists() or RECEIPT_PATH.exists():
        raise CorrectionRunnerError("Correction intent/output namespace must be empty before preparation")
    auth, freeze, train_path = _load_authority(verify_committed=True)
    historical_hashes = _verify_historical_artifacts(auth, verify_committed=True)
    predictions = _load_predictions(auth)
    selection = core.selection_binding(predictions)
    if selection["selected_rows_count"] != 20_903:
        raise CorrectionRunnerError("Selected-label intent is not exactly 20,903 rows")
    if not SYNTHETIC_QA_PATH.is_file() or _read_json(SYNTHETIC_QA_PATH).get("status") != "PASS":
        raise CorrectionRunnerError("Passing synthetic correction QA is required before intent")
    current_sources = _source_hashes()
    body = {
        "schema_version": 1, "correction_id": core.CORRECTION_ID,
        "status": "AUTHORIZED_NOT_READ",
        "authorization": {"path": str(AUTH_PATH.relative_to(ROOT).as_posix()),
                           "sha256": _sha256_file(AUTH_PATH),
                           "status": auth["status"],
                           "additional_fits_allowed": 0,
                           "additional_selected_label_reads_allowed": 1},
        "historical_invalid_outputs_preserved": historical_hashes,
        "source_freeze_signature": freeze["freeze_signature"],
        "official_train": {"path": str(train_path),
                            "sha256": freeze["official_source"]["sha256"],
                            "rows": int(freeze["official_source"]["rows"])},
        "prediction_hashes": auth["prediction_hashes"],
        "selected_primary_rows": {
            "count": selection["selected_rows_count"],
            "rows_sha256": selection["selected_rows_sha256"],
            "record_ids_sha256": selection["selected_ids_sha256"],
            "origins": selection["origins"],
            "row_order": "source Train row order ascending",
            "target_columns_selected": ["record_id", "artifact_signal"],
            "target_value_read_method": "one pandas CSV selection with skiprows on nonselected data rows; usecols contains only record_id and artifact_signal",
        },
        "read_accounting_before_correction": {
            "original_evaluation_target_column_reads": int(auth["original_actual_target_column_read_count"]),
            "original_target_values_materialized": int(auth["original_actual_target_vector_rows"]),
            "original_unique_primary_rows_scored": 20_903,
            "original_nonprimary_target_values_also_materialized":
                int(auth["original_actual_target_vector_rows"]) - 20_903,
        },
        "candidate_or_policy_changes_allowed": False,
        "additional_predictor_or_representation_fits_allowed": 0,
        "query_target_values_materialized_before_read": 0,
        "source_hashes": current_sources,
        "synthetic_qa_sha256": _sha256_file(SYNTHETIC_QA_PATH),
        "test_or_sample_opened": False,
        "created_at_utc": _utc_now(),
    }
    body["intent_signature"] = _canonical_sha256(body)
    _atomic_json(INTENT_PATH, body)
    print(json.dumps({"status": body["status"], "path": str(INTENT_PATH),
                      "sha256": _sha256_file(INTENT_PATH),
                      "selected_rows": selection["selected_rows_count"],
                      "selected_ids_sha256": selection["selected_ids_sha256"],
                      "query_target_values_materialized": 0,
                      "source_paths": len(current_sources)}, sort_keys=True))


def _check_committed_source_set(intent: Mapping[str, Any]) -> None:
    for relative, expected in intent["source_hashes"].items():
        path = ROOT / relative
        _assert_committed_exact(path, relative)
        if _sha256_file(path) != expected:
            raise CorrectionRunnerError(f"Correction source changed after intent commit: {relative}")
    _assert_committed_exact(SYNTHETIC_QA_PATH)
    if _sha256_file(SYNTHETIC_QA_PATH) != intent["synthetic_qa_sha256"]:
        raise CorrectionRunnerError("Committed correction synthetic QA differs from intent")
    _assert_committed_exact(INTENT_PATH)


def _read_selected_labels_once(train_path: Path, selection: Mapping[str, Any], *,
                               expected_rows: int = 20_903) -> tuple[np.ndarray, np.ndarray]:
    selected_rows = np.asarray(selection["selected_rows"], dtype=np.int64)
    selected_ids = np.asarray(selection["selected_ids"]).astype(str)
    selected_set = set(int(row) for row in selected_rows.tolist())

    def skip_line(line_number: int) -> bool:
        if line_number == 0:
            return False
        return (line_number - 1) not in selected_set

    # This is the single authorized query-target value read. Nonselected data rows are skipped.
    frame = pd.read_csv(train_path, usecols=["record_id", "artifact_signal"],
                        dtype={"record_id": "string"}, skiprows=skip_line)
    if len(frame) != len(selected_rows) or len(frame) != expected_rows:
        raise CorrectionRunnerError(f"Selected Train read did not return exactly {expected_rows} rows")
    observed_ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    if not np.array_equal(observed_ids, selected_ids):
        raise CorrectionRunnerError("Selected Train IDs do not match frozen prediction IDs in source-row order")
    labels = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int64)
    if not np.isin(labels, (0, 1)).all():
        raise CorrectionRunnerError("Selected correction target values are not binary 0/1")
    return observed_ids, labels


def _build_report(results: Mapping[str, Any]) -> str:
    lines = [
        "# CLS_PPMI_CONTEXT_V1 corrected scoring — CORRECTION_R1", "",
        f"Corrected gate disposition: **{results['corrected_metrics']['overall']}**.", "",
        "This additive result uses the committed frozen predictions and the intended MacroF1, temporal-view, all-view subgroup, and mechanism contracts. The earlier report remains unchanged and is invalid for that intended contract.", "",
        "## MacroF1 views", "",
        "| View | Rows | Champion | Real candidate | Shuffled control | Real − champion | Real − shuffled |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, view in results["corrected_metrics"]["views"].items():
        metrics = view["metrics"]
        gains = metrics["gains"]
        lines.append(
            f"| {name} | {view['rows']} | {metrics['champion']['macro_f1']:.9f} | "
            f"{metrics['candidate_real']['macro_f1']:.9f} | {metrics['control_shuffled']['macro_f1']:.9f} | "
            f"{gains['candidate_real_minus_champion_macro_f1']:+.9f} | "
            f"{gains['candidate_real_minus_control_shuffled_macro_f1']:+.9f} |"
        )
    gates = results["corrected_metrics"]["gates"]
    scientific = gates["scientific"]
    lines += [
        "", "## Gates", "",
        f"- Scientific: **{scientific['status']}**.",
        f"- Exploratory: **{gates['exploratory']['status']}**.",
        f"- Mechanism: **{gates['mechanism']['status']}**.",
        f"- Worst supported MacroF1 harm across all six views: {scientific['worst_supported_macro_f1_harm_all_views']:.9f}.",
        f"- Worst supported positive-recall harm across all six views: {scientific['worst_supported_positive_recall_harm_all_views']:.9f}.",
        f"- Supported protected groups by view: `{json.dumps(scientific['supported_groups_by_view'], sort_keys=True)}`.",
        "", "## Fit and representation audit", "",
        "The four existing fits were not rerun. The frozen prefix, tokenizer/vocabulary, PPMI, SVD, HGB parameters, saved model/cache identities, and runtime artifact hashes were audited against the root campaign; details are in `RECIPE_AUDIT.md`.",
        "Training fit F1 diagnostics are corrected from the immutable confusion matrices without reading training labels again.",
        "", "## Label-read accounting", "",
        f"- Earlier scoring read {results['read_accounting']['original_target_values_materialized']} target values in one full-column read; only {results['read_accounting']['original_unique_primary_rows_scored']} unique primary rows were scored.",
        f"- This correction performed one selected read of {results['read_accounting']['correction_target_values_materialized']} target values.",
        f"- Cumulative evaluation target values materialized across both passes: {results['read_accounting']['cumulative_target_values_materialized']}; the independent metric verifier made {results['read_accounting']['additional_verifier_target_reads']} target reads.",
        "- `Test.csv` and `SampleSubmission.csv` were not opened. No fit, threshold, blend, or candidate change was made.",
        "", "## Decision boundary", "",
        "This report corrects the earlier scoring contract only. The root retains the family disposition and next-lane decision; no new fit is authorized by this correction.",
        "",
    ]
    return "\n".join(lines)


def command_score() -> None:
    if not INTENT_PATH.is_file() or not _check_signature(_read_json(INTENT_PATH), "intent_signature"):
        raise CorrectionRunnerError("Committed correction intent/signature is required")
    intent = _read_json(INTENT_PATH)
    if ATTEMPT_PATH.exists() or any(path.exists() for path in
                                    (RESULTS_PATH, REPORT_PATH, MEMORY_VERIFICATION_PATH, RECEIPT_PATH)):
        raise CorrectionRunnerError("One-shot correction label read was already attempted; refusing retry")
    _check_committed_source_set(intent)
    auth, freeze, train_path = _load_authority(verify_committed=True)
    historical_hashes = _verify_historical_artifacts(auth, verify_committed=True)
    if historical_hashes != intent["historical_invalid_outputs_preserved"]:
        raise CorrectionRunnerError("Historical immutable artifacts differ from committed correction intent")
    predictions = _load_predictions(auth)
    selection = core.selection_binding(predictions)
    if selection["selected_rows_sha256"] != intent["selected_primary_rows"]["rows_sha256"] \
            or selection["selected_ids_sha256"] != intent["selected_primary_rows"]["record_ids_sha256"]:
        raise CorrectionRunnerError("Selected row/ID bindings differ from the pre-read intent")
    if _sha256_file(train_path) != freeze["official_source"]["sha256"]:
        raise CorrectionRunnerError("Frozen Train bytes changed before the selected one-read correction")
    # A write-once start marker consumes the one authorized attempt if the process later fails.
    attempt = {
        "schema_version": 1, "correction_id": core.CORRECTION_ID,
        "status": "ONE_SELECTED_READ_ATTEMPT_STARTED",
        "intent_signature": intent["intent_signature"],
        "source_sha256": freeze["official_source"]["sha256"],
        "selected_rows": selection["selected_rows_count"],
        "selected_rows_sha256": selection["selected_rows_sha256"],
        "selected_ids_sha256": selection["selected_ids_sha256"],
        "target_values_expected": 20_903,
        "selected_read_call_count": 1,
        "created_at_utc": _utc_now(),
    }
    _atomic_json(ATTEMPT_PATH, attempt)
    # From this point onward, no command in this correction namespace may reopen target labels.
    observed_ids, selected_labels = _read_selected_labels_once(train_path, selection)
    labels_by_id = core.labels_to_map(observed_ids, selected_labels)
    label_vector_sha256 = hashlib.sha256(np.ascontiguousarray(selected_labels.astype(np.uint8)).tobytes()).hexdigest()
    prelabel_sha = historical_hashes["prelabel_verification"]
    corrected_metrics = core.score_correction(predictions, labels_by_id,
                                              prelabel_verification_sha256=prelabel_sha)
    anchors = auth["no_label_root_checks"]["confusion_matrix_only_macro_reconstruction"]
    independent = verify_in_memory(predictions, labels_by_id, corrected_metrics, anchors)
    fit_summary = _read_json(FIT_SUMMARY_PATH)
    corrected_train_diagnostics = core.training_macro_f1_readback(fit_summary)
    body = {
        "schema_version": 1, "correction_id": core.CORRECTION_ID,
        "authorization_sha256": _sha256_file(AUTH_PATH),
        "authorization_status": auth["status"],
        "original_output_disposition": auth["original_verdict_status"],
        "original_artifacts_sha256": historical_hashes,
        "source_freeze_signature": freeze["freeze_signature"],
        "original_prediction_hashes": auth["prediction_hashes"],
        "query_target_column": "artifact_signal",
        "query_target_values_read": 20_903,
        "query_target_selected_row_ids_sha256": selection["selected_ids_sha256"],
        "query_target_selected_source_rows_sha256": selection["selected_rows_sha256"],
        "query_target_vector_sha256": label_vector_sha256,
        "read_accounting": {
            "original_target_column_reads": int(auth["original_actual_target_column_read_count"]),
            "original_target_values_materialized": int(auth["original_actual_target_vector_rows"]),
            "original_unique_primary_rows_scored": 20_903,
            "original_nonprimary_target_values_also_materialized":
                int(auth["original_actual_target_vector_rows"]) - 20_903,
            "correction_target_column_reads": 1,
            "correction_target_values_materialized": 20_903,
            "cumulative_target_column_reads": int(auth["original_actual_target_column_read_count"]) + 1,
            "cumulative_target_values_materialized": int(auth["original_actual_target_vector_rows"]) + 20_903,
            "unique_source_rows_across_both_passes": 50_608,
            "selected_rows_reloaded_by_correction": 20_903,
            "additional_verifier_target_reads": 0,
            "target_values_stored_in_result": False,
        },
        "fit_accounting": {"additional_fits": 0, "original_representation_fits": 4,
                           "original_predictor_fits": 4, "original_fit_retries": 0},
        "corrected_metrics": corrected_metrics,
        "corrected_training_fit_diagnostics_from_frozen_confusion": corrected_train_diagnostics,
        "independent_verification": independent,
        "test_or_sample_opened": False,
        "family_or_next_lane_decision": "ROOT_REVIEW_REQUIRED",
        "created_at_utc": _utc_now(),
    }
    body["correction_signature"] = _canonical_sha256(body)
    report = _build_report(body)
    _atomic_json(RESULTS_PATH, body)
    _atomic_write(REPORT_PATH, report.encode("utf-8"))
    memory_verification = {
        "schema_version": 1, "correction_id": core.CORRECTION_ID,
        "status": independent["status"], "independent_verification": independent,
        "selected_target_rows_in_memory": len(labels_by_id),
        "selected_target_vector_sha256": label_vector_sha256,
        "target_read_calls_in_verifier": 0,
        "created_at_utc": _utc_now(),
    }
    memory_verification["verification_signature"] = _canonical_sha256(memory_verification)
    _atomic_json(MEMORY_VERIFICATION_PATH, memory_verification)
    receipt = {
        "schema_version": 1, "correction_id": core.CORRECTION_ID,
        "status": "PASS", "intent_signature": intent["intent_signature"],
        "source_freeze_signature": freeze["freeze_signature"],
        "corrected_result_sha256": _sha256_file(RESULTS_PATH),
        "corrected_report_sha256": _sha256_file(REPORT_PATH),
        "in_memory_verification_sha256": _sha256_file(MEMORY_VERIFICATION_PATH),
        "score_read_attempt_sha256": _sha256_file(ATTEMPT_PATH),
        "target_read_accounting": body["read_accounting"],
        "independent_verifier_status": independent["status"],
        "query_targets_read_once_after_committed_intent": True,
        "target_values_stored": False,
        "old_result_report_postlabel_receipt_and_predictions_unchanged": True,
        "created_at_utc": _utc_now(),
    }
    receipt["receipt_signature"] = _canonical_sha256(receipt)
    _atomic_json(RECEIPT_PATH, receipt)
    print(json.dumps({"status": receipt["status"],
                      "overall": corrected_metrics["overall"],
                      "path": str(RESULTS_PATH),
                      "query_target_values_read": 20_903,
                      "additional_verifier_target_reads": 0,
                      "primary_macro_f1_gain": corrected_metrics["views"]["PRIMARY_UNIQUE"]["metrics"]["gains"]["candidate_real_minus_champion_macro_f1"],
                      "scientific_gate": corrected_metrics["gates"]["scientific"]["status"],
                      "exploratory_gate": corrected_metrics["gates"]["exploratory"]["status"]}, sort_keys=True))


def command_verify() -> None:
    auth, freeze, _ = _load_authority(verify_committed=True)
    historical_hashes = _verify_historical_artifacts(auth, verify_committed=True)
    for path in (INTENT_PATH, ATTEMPT_PATH, RESULTS_PATH, REPORT_PATH,
                 MEMORY_VERIFICATION_PATH, RECEIPT_PATH):
        _assert_committed_exact(path)
    intent = _read_json(INTENT_PATH)
    results = _read_json(RESULTS_PATH)
    memory_verification = _read_json(MEMORY_VERIFICATION_PATH)
    receipt = _read_json(RECEIPT_PATH)
    attempt = _read_json(ATTEMPT_PATH)
    _check_committed_source_set(intent)
    if not _check_signature(intent, "intent_signature") \
            or not _check_signature(results, "correction_signature") \
            or not _check_signature(memory_verification, "verification_signature") \
            or not _check_signature(receipt, "receipt_signature"):
        raise CorrectionRunnerError("Correction artifact signature mismatch")
    if intent.get("source_freeze_signature") != freeze["freeze_signature"] \
            or results.get("source_freeze_signature") != freeze["freeze_signature"] \
            or receipt.get("source_freeze_signature") != freeze["freeze_signature"]:
        raise CorrectionRunnerError("Correction artifacts are not bound to the frozen source")
    accounting = results.get("read_accounting", {})
    expected_accounting = {
        "original_target_column_reads": 1,
        "original_target_values_materialized": 50_608,
        "original_unique_primary_rows_scored": 20_903,
        "original_nonprimary_target_values_also_materialized": 29_705,
        "correction_target_column_reads": 1,
        "correction_target_values_materialized": 20_903,
        "cumulative_target_column_reads": 2,
        "cumulative_target_values_materialized": 71_511,
        "unique_source_rows_across_both_passes": 50_608,
        "selected_rows_reloaded_by_correction": 20_903,
        "additional_verifier_target_reads": 0,
        "target_values_stored_in_result": False,
    }
    if accounting != expected_accounting or receipt.get("target_read_accounting") != expected_accounting:
        raise CorrectionRunnerError("Final target-value read accounting differs from authorization")
    if int(attempt.get("target_values_expected", -1)) != 20_903 \
            or int(attempt.get("selected_read_call_count", -1)) != 1 \
            or memory_verification.get("target_read_calls_in_verifier") != 0 \
            or receipt.get("independent_verifier_status") != "PASS":
        raise CorrectionRunnerError("One-read / no-extra-read verification receipt failed")
    if historical_hashes != intent.get("historical_invalid_outputs_preserved"):
        raise CorrectionRunnerError("A pinned historical output changed after correction")
    if results.get("original_output_disposition") != auth["original_verdict_status"] \
            or results.get("family_or_next_lane_decision") != "ROOT_REVIEW_REQUIRED":
        raise CorrectionRunnerError("Correction misstates historical result or root decision boundary")
    if results.get("corrected_metrics", {}).get("overall") not in {
            "SCIENTIFIC_PASS", "EXPLORATORY_ONLY_ELIGIBLE", "NO_ADVANCE"}:
        raise CorrectionRunnerError("Corrected metric disposition is missing")
    if results.get("test_or_sample_opened") is not False:
        raise CorrectionRunnerError("Correction receipt claims Test/Sample access")
    if _sha256_file(RESULTS_PATH) != receipt["corrected_result_sha256"] \
            or _sha256_file(REPORT_PATH) != receipt["corrected_report_sha256"] \
            or _sha256_file(MEMORY_VERIFICATION_PATH) != receipt["in_memory_verification_sha256"] \
            or _sha256_file(ATTEMPT_PATH) != receipt["score_read_attempt_sha256"]:
        raise CorrectionRunnerError("Correction output hash differs from final receipt")
    print(json.dumps({"status": "PASS", "phase": "committed_structural_no_label_reread",
                      "overall": results["corrected_metrics"]["overall"],
                      "results_sha256": _sha256_file(RESULTS_PATH),
                      "historical_hashes_unchanged": True,
                      "query_target_reads_by_verifier": 0,
                      "cumulative_target_values_materialized": 71_511}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("synthetic", help="Run isolated corrected scoring lifecycle QA")
    sub.add_parser("prepare-intent", help="Bind correction intent without target labels")
    sub.add_parser("score", help="Consume exactly one authorized selected target vector")
    sub.add_parser("verify", help="Verify committed outputs without rereading labels")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            command_synthetic()
        elif args.command == "prepare-intent":
            command_prepare_intent()
        elif args.command == "score":
            command_score()
        elif args.command == "verify":
            command_verify()
        else:
            parser.error("unhandled command")
        return 0
    except (CorrectionRunnerError, core.CorrectionError,
            IndependentVerificationError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
