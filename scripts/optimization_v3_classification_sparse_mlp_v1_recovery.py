"""Narrow recovery amendment for the MLP post-fit serialization failure.

This module does not refit cutoff 55. It reconstructs a missing JSON receipt
and prediction bundle from the estimator and preprocessing object that were
durably saved before the original script failed. For cutoff 67 it supplies a
JSON-safe fit-state serializer to the original frozen runner at runtime; the
recipe, model, and original frozen source files remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from scripts import optimization_v3_classification_sparse_mlp_v1 as runner
from src.models.classification_models import _array_digest, _json_digest
from src.models.classification_sparse_mlp_v1 import transform_score_design
from src.models.research_next_budget import _append, read_events
from src.models.research_next_contract import publish_bundle, read_bundle
from src.models.round1_models import _to_builtin


RECOVERY_NOTE = runner.REPORT_ROOT / "POSTFIT_SERIALIZATION_RECOVERY.json"


def _vocab_sha(vectorizer: Any) -> str:
    vocab = {} if vectorizer is None else {str(key): int(value) for key, value in vectorizer.vocabulary_.items()}
    return _json_digest(vocab)


def _fit_state_json_safe(base: Any, estimator: Any, fit_matrix: Any, target: np.ndarray,
                         cutoff: int, warnings_seen: list[str], seconds: float | None,
                         model_sha: str) -> dict[str, Any]:
    """Same audit fields as the frozen serializer with safe NumPy scalar handling."""
    vectorizer = base._text_vectorizer
    idf_sha = hashlib.sha256(np.asarray([], dtype="<f8").tobytes() if vectorizer is None else
                             np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest()
    matrix_sha = hashlib.sha256(fit_matrix.indptr.tobytes() + fit_matrix.indices.tobytes() + fit_matrix.data.tobytes()).hexdigest()
    target_sha = hashlib.sha256(np.asarray(target, dtype=np.int8).tobytes()).hexdigest()
    return {
        "status": "MODEL_STATE_DURABLE",
        "card_id": runner.CARD_ID, "stage": runner.STAGE, "cutoff": int(cutoff),
        "fit_rows": int(len(target)), "fit_cycle_min": 47, "fit_cycle_max": int(cutoff),
        "training_design_shape": [int(x) for x in fit_matrix.shape], "training_design_nnz": int(fit_matrix.nnz),
        "training_design_sha256": matrix_sha, "target_sha256": target_sha,
        "preprocessing_state": {
            "recipe": "exact C2_B structured numeric imputation/scaling + OHE + TF-IDF 1-2 grams + stream missing indicator",
            "fit_rows_only": int(len(target)), "feature_names": list(base.input_feature_names),
            "structured_feature_count_including_missing_indicator": int(base._transform_state["structured_columns"]),
            "tfidf_feature_count": int(base._transform_state["text_columns"]),
            "tfidf_vocabulary_size": 0 if vectorizer is None else int(len(vectorizer.vocabulary_)),
            "tfidf_vocabulary_sha256": _vocab_sha(vectorizer), "tfidf_idf_sha256": idf_sha,
            "structured_preprocessor": base._structured_preprocessor.audit_state(),
            "c2b_historical_preprocessing_parity": getattr(base, "_mlp_c2b_preprocessing_parity", None),
        },
        "estimator_parameters": _to_builtin(estimator.get_params(deep=True)),
        "n_iter": int(estimator.n_iter_), "loss_curve": [float(x) for x in estimator.loss_curve_],
        "initial_training_loss": float(estimator.loss_curve_[0]),
        "final_training_loss": float(estimator.loss_curve_[-1]),
        "convergence_warnings": warnings_seen, "convergence_certified": False,
        "fit_seconds": None if seconds is None else float(seconds),
        "fit_seconds_status": "NOT_PERSISTED_BEFORE_SERIALIZATION_FAILURE" if seconds is None else "RECORDED",
        "model_joblib_sha256": model_sha,
        "score_labels_passed_to_fit": False, "test_used": False,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Refusing to overwrite nonidentical recovery receipt: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _recovered_event(fit_id: str, cutoff: int, receipt_sha256: str,
                    model_sha: str, state_sha: str) -> dict[str, Any]:
    events = read_events(runner.LEDGER)
    existing = [event for event in events if event.get("event") == "FIT_OUTPUT_RECOVERED"
                and event.get("fit_id") == fit_id]
    payload = {"event": "FIT_OUTPUT_RECOVERED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": runner.CARD_ID, "stage": runner.STAGE, "fit_id": fit_id,
        "cutoff": int(cutoff), "original_terminal_status": "FIT_FAILED",
        "original_terminal_event_preserved": True, "refit_performed": False,
        "model_joblib_sha256": model_sha, "fit_state_sha256": state_sha,
        "bundle_receipt_sha256": receipt_sha256,
        "recovery_code_sha256": _sha(Path(__file__))}
    if existing:
        if len(existing) != 1 or any(existing[0].get(key) != value for key, value in payload.items() if key != "event"):
            raise RuntimeError(f"Conflicting duplicate artifact recovery event for {fit_id}")
        return existing[0]
    return _append(runner.LEDGER, payload)


def _sha(path: Path) -> str:
    return runner._sha(path)


def recover_cutoff55() -> dict[str, Any]:
    frozen = runner._check_preflight()
    cutoff = 55
    fit_id = f"{runner.CARD_ID}_{cutoff}"
    events = read_events(runner.LEDGER)
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    failures = [event for event in events if event.get("event") == "FIT_FAILED" and event.get("fit_id") == fit_id]
    if len(starts) != 1 or len(failures) != 1:
        raise RuntimeError("Cutoff 55 recovery requires exactly one preserved FIT_STARTED/FIT_FAILED pair")
    if "int64 is not JSON serializable" not in str(failures[0].get("error")):
        raise RuntimeError("Cutoff 55 failed for an unexpected reason; do not apply this recovery")

    fit_dir = runner.RUN_ROOT / f"fit{cutoff}"
    model_path = fit_dir / "fitted_state.joblib"
    state_path = fit_dir / "fit_state.json"
    bundle_path = fit_dir / "bundle"
    if not model_path.is_file() or state_path.exists() or bundle_path.exists():
        raise RuntimeError("Unexpected partial cutoff 55 state; inspect before recovery")
    model_sha = _sha(model_path)
    stored = joblib.load(model_path)
    base, estimator = stored["base"], stored["estimator"]
    if int(estimator.n_iter_) != 30 or len(estimator.loss_curve_) != 30:
        raise RuntimeError("Durable cutoff 55 MLP is not the frozen 30-epoch fit")
    if not np.isfinite(estimator.loss_curve_).all() or not np.array_equal(estimator.classes_, np.asarray([0, 1])):
        raise RuntimeError("Durable cutoff 55 estimator state is invalid")

    context = runner._load_context()
    rebuilt_base, fit_matrix, y_fit, _score_matrix_unused, score_rows = runner._design_for_origin(context, cutoff)
    parity_saved = runner._verify_c2b_preprocessing(base, cutoff)
    parity_rebuilt = runner._verify_c2b_preprocessing(rebuilt_base, cutoff)
    if parity_saved["status"] != "PASS" or parity_rebuilt["status"] != "PASS":
        raise RuntimeError("Durable cutoff 55 preprocessing does not match historical C2_B audit")
    if not np.array_equal(score_rows, context["baselines"][cutoff]["rows"]):
        raise RuntimeError("Cutoff 55 score rows changed during recovery")

    state = _fit_state_json_safe(base, estimator, fit_matrix, y_fit, cutoff,
                                 ["Original serializer failed after fitted_state.joblib was durable; fixed JSON-safe recovery serializer."],
                                 None, model_sha)
    state["freeze_signature"] = frozen["signature"]
    state["input_hashes"] = frozen["identity"]["input_sha256"]
    state["recovery"] = {"type": "POSTFIT_SERIALIZATION_RECOVERY", "refit_performed": False,
        "original_fit_terminal_event": "FIT_FAILED", "original_fit_error": failures[0]["error"],
        "preprocessing_parity_historical_C2_B": parity_saved["checks"],
        "preprocessing_parity_rebuilt_C2_B": parity_rebuilt["checks"],
        "fit_seconds_unavailable": True}
    _atomic_json(state_path, state)
    state_sha = _sha(state_path)

    frame = context["frame"]
    baseline = context["baselines"][cutoff]
    feature_names = [name for name in frame.columns if name not in {"record_id", "artifact_signal"}]
    score_x = frame.iloc[score_rows].loc[:, feature_names]
    score_matrix = transform_score_design(base, score_x)
    probability = np.asarray(estimator.predict_proba(score_matrix), dtype=np.float64)
    if probability.shape != (len(score_rows), 2) or not np.isfinite(probability).all():
        raise RuntimeError("Recovered cutoff 55 MLP probabilities are invalid")
    raw = probability[:, 1]
    missing = np.asarray(baseline["missing"], dtype=bool)
    candidate = runner.route_probabilities("candidate", baseline["baseline"], raw,
        baseline["c2b"], baseline["cb"], missing)
    matched = runner.route_probabilities("matched_control", baseline["baseline"], raw,
        baseline["c2b"], baseline["cb"], missing)
    payload = {
        "score_rows": score_rows.astype(np.int64),
        "score_record_ids": frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str),
        "cycle_code": context["cycles"][score_rows].astype(np.int16),
        "prediction": candidate.astype(np.float64),
        "baseline_prediction": np.asarray(baseline["baseline"], dtype=np.float64),
        "raw_mlp_probability": raw,
        "matched_prediction": matched.astype(np.float64),
        "control_c2b_probability": np.asarray(baseline["c2b"], dtype=np.float64),
        "control_cb_probability": np.asarray(baseline["cb"], dtype=np.float64),
        "control_stream_missing": missing,
        "payload_meta": {
            "card_id": runner.CARD_ID, "stage": runner.STAGE, "cutoff": cutoff,
            "probability_semantics": "positive-class probability; prediction is final routed probability",
            "baseline_probability_semantics": "frozen sparse25 final routed probability",
            "candidate_route": "present=.75*sparse25+.25*raw_MLP; missing=frozen-CB; threshold=.50",
            "matched_control_route": "present=.75*sparse25+.25*C2_B; missing=frozen-CB; threshold=.50",
            "test_used": False, "score_labels_passed_to_fit": False,
            "recovered_after_fit_serializer_error": failures[0]["error"],
            "refit_performed": False, "model_joblib_sha256": model_sha,
            "fit_state_sha256": state_sha,
        },
    }
    bundle_identity = runner._bundle_identity(frozen["identity"], frozen["signature"], cutoff, model_sha, state_sha)
    receipt = publish_bundle(bundle_path, payload, bundle_identity, task="classification")
    saved = read_bundle(bundle_path, bundle_identity, task="classification")
    rebuilt_route = runner.route_probabilities("candidate", saved.arrays["baseline_prediction"],
        saved.arrays["raw_mlp_probability"], saved.arrays["control_c2b_probability"],
        saved.arrays["control_cb_probability"], saved.arrays["control_stream_missing"])
    np.testing.assert_array_equal(rebuilt_route, saved.arrays["prediction"])
    if saved.receipt["receipt_sha256"] != receipt["receipt_sha256"]:
        raise RuntimeError("Recovered cutoff 55 bundle did not read back exactly")
    event = _recovered_event(fit_id, cutoff, receipt["receipt_sha256"], model_sha, state_sha)
    note = {
        "status": "RECOVERED_NO_REFIT", "fit_id": fit_id, "cutoff": cutoff,
        "original_fit_terminal_event": "FIT_FAILED", "original_fit_terminal_error": failures[0]["error"],
        "original_budget_start_utc": starts[0].get("at_utc"),
        "recovery_event": event, "model_joblib_sha256": model_sha,
        "fit_state_sha256": state_sha, "bundle_receipt_sha256": receipt["receipt_sha256"],
        "refit_performed": False, "score_rows": int(len(score_rows)),
        "train_sha256": _sha(context["train_path"]), "test_read": False,
        "sample_submission_read": False, "kaggle_access": False,
    }
    _atomic_json(RECOVERY_NOTE, note)
    return note


def run_cutoff67_with_safe_serializer() -> dict[str, Any]:
    frozen = runner._check_preflight()
    fit_id = f"{runner.CARD_ID}_67"
    starts = [event for event in read_events(runner.LEDGER)
              if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    if starts:
        raise RuntimeError("Cutoff 67 already has a FIT_STARTED event; do not start another fit")
    _mlp55_terminal = [event for event in read_events(runner.LEDGER)
                       if event.get("event") in {"FIT_FAILED", "FIT_SAVED"}
                       and event.get("fit_id") == f"{runner.CARD_ID}_55"]
    if len(_mlp55_terminal) != 1 or _mlp55_terminal[0].get("event") != "FIT_FAILED":
        raise RuntimeError("Cutoff 67 recovery wrapper expected the preserved cutoff 55 FIT_FAILED event")
    previous = runner._fit_state
    runner._fit_state = _fit_state_json_safe
    try:
        result = runner.fit_origin(67)
    finally:
        runner._fit_state = previous
    return {"result": result, "recovery_code_sha256": _sha(Path(__file__)),
            "freeze_signature": frozen["signature"], "recipe_unchanged": True}


def _outputs(context: Mapping[str, Any], identity: Mapping[str, Any], signature: str) -> dict[int, dict[str, Any]]:
    outputs: dict[int, dict[str, Any]] = {}
    for cutoff in runner.ORIGINS:
        fit_dir = runner.RUN_ROOT / f"fit{cutoff}"
        model_path = fit_dir / "fitted_state.joblib"
        state_path = fit_dir / "fit_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        bundle_identity = runner._bundle_identity(identity, signature, cutoff, _sha(model_path), _sha(state_path))
        bundle = read_bundle(fit_dir / "bundle", bundle_identity, task="classification")
        arrays = bundle.arrays
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        expected_rows = context["baselines"][cutoff]["rows"]
        if not np.array_equal(rows, expected_rows):
            raise RuntimeError(f"Saved MLP bundle rows differ at cutoff {cutoff}")
        expected_ids = context["frame"].iloc[rows]["record_id"].astype(str).to_numpy()
        if not np.array_equal(arrays["score_record_ids"].astype(str), expected_ids):
            raise RuntimeError(f"Saved MLP bundle IDs differ at cutoff {cutoff}")
        outputs[cutoff] = {key: arrays[key] for key in arrays}
        outputs[cutoff].update({"origin": np.full(len(rows), cutoff, dtype=np.int64),
            "cycle_code": context["cycles"][rows], "loss_curve": state["loss_curve"],
            "fit_state": state, "receipt": bundle.receipt,
            "c2b_diagnostic": np.where(arrays["control_stream_missing"].astype(bool),
                arrays["control_cb_probability"], arrays["control_c2b_probability"])})
    return outputs


def finalize_and_verify() -> dict[str, Any]:
    from scripts.optimization_v3_classification_sparse_mlp_v1_recovery_verify import verify_recovered

    frozen = runner._check_preflight()
    context = runner._load_context()
    outputs = _outputs(context, frozen["identity"], frozen["signature"])
    metrics, groups, per_cycle, gate = runner.evaluate_views(outputs, context["frame"], context["target"])
    release_event = runner.release_unused_reservation()
    budget_snapshot = {"stage": runner.STAGE, "release_event": release_event,
        "ledger_summary": runner.ledger_summary(runner.LEDGER)}
    _atomic_json(runner.REPORT_ROOT / "budget_snapshot.json", budget_snapshot)
    events = read_events(runner.LEDGER)
    fit_status = {}
    for cutoff in runner.ORIGINS:
        fit_id = f"{runner.CARD_ID}_{cutoff}"
        terminal = [event for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                    and event.get("fit_id") == fit_id]
        recovered = [event for event in events if event.get("event") == "FIT_OUTPUT_RECOVERED"
                     and event.get("fit_id") == fit_id]
        if cutoff == 55:
            if len(terminal) != 1 or terminal[0].get("event") != "FIT_FAILED" or len(recovered) != 1:
                raise RuntimeError("Cutoff 55 failure/recovery ledger chain is inconsistent")
            fit_status[str(cutoff)] = {"status": "FIT_FAILED_OUTPUT_RECOVERED_NO_REFIT",
                "original_terminal_error": terminal[0].get("error"),
                "recovery_receipt_sha256": recovered[0].get("bundle_receipt_sha256"),
                "estimator_fit_consumed": True}
        else:
            if len(terminal) != 1 or terminal[0].get("event") != "FIT_SAVED":
                raise RuntimeError("Cutoff 67 must have exactly one FIT_SAVED ledger event")
            fit_status[str(cutoff)] = {"status": "FIT_SAVED",
                "receipt_sha256": terminal[0].get("receipt_sha256"), "estimator_fit_consumed": True}
    fits = {str(cutoff): {"ledger_status": fit_status[str(cutoff)]["status"],
        "fit_rows": outputs[cutoff]["fit_state"]["fit_rows"],
        "score_rows": int(len(outputs[cutoff]["score_rows"])),
        "n_iter": outputs[cutoff]["fit_state"]["n_iter"],
        "loss_curve": outputs[cutoff]["loss_curve"],
        "model_joblib_sha256": outputs[cutoff]["fit_state"]["model_joblib_sha256"],
        "fit_state_sha256": _sha(runner.RUN_ROOT / f"fit{cutoff}/fit_state.json"),
        "bundle_receipt_sha256": outputs[cutoff]["receipt"]["receipt_sha256"],
        "fit_seconds": outputs[cutoff]["fit_state"]["fit_seconds"],
        "fit_seconds_status": outputs[cutoff]["fit_state"]["fit_seconds_status"],
        "convergence_warnings": outputs[cutoff]["fit_state"]["convergence_warnings"]}
        for cutoff in runner.ORIGINS}
    result = {
        "status": "COMPLETE_REVIEW_REQUIRED", "campaign": runner.CARD_ID,
        "scientific_verdict": gate["status"], "freeze_signature": frozen["signature"],
        "source_commit": frozen["identity"]["source_commit"],
        "official_train_sha256": _sha(context["train_path"]),
        "official_train_rows": int(len(context["frame"])), "official_fits_started": 2,
        "official_fits_with_durable_estimators": 2, "official_cutoffs": fits,
        "candidate_policy": "present=.75*sparse25+.25*MLP; missing=frozen-CB; threshold=.50",
        "matched_control_policy": "present=.75*sparse25+.25*C2_B; missing=frozen-CB; threshold=.50",
        "metrics_by_view": metrics, "protected_groups": groups, "per_cycle": per_cycle,
        "frozen_gate": gate,
        "fit_recovery": {
            "cutoff55": "Estimator and preprocessing were durable before JSON serializer TypeError; outputs were rebuilt from the saved estimator without refitting.",
            "ledger_history_preserved": True, "fit55_terminal_event_remains_FIT_FAILED": True,
            "fit67_serializer_patch_applied_before_fit": True,
            "recovery_code_sha256": _sha(Path(__file__)),
            "recovery_note_sha256": _sha(RECOVERY_NOTE),
        },
        "budget": budget_snapshot["ledger_summary"],
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
        "interpretation_limits": [
            "Two temporal origins are development validation, not an untouched holdout.",
            "The matched policy controls allocation reduction but does not causally isolate nonlinear capacity from optimizer/model differences.",
            "Thirty epochs are fixed by protocol; convergence is not certified.",
            "Cutoff 55's original ledger terminal status remains FIT_FAILED for a post-fit serialization error; durable estimator outputs were independently recovered without refit.",
            "No Test, SampleSubmission, full-Train refit, CSV, or Kaggle access occurred.",
        ],
    }
    _atomic_json(runner.REPORT_ROOT / "results.json", result)
    runner._write_csv_once(runner.REPORT_ROOT / "protected_groups.csv", groups)
    runner._write_csv_once(runner.REPORT_ROOT / "per_cycle.csv", per_cycle)
    verification = verify_recovered()
    if verification.get("status") != "PASS":
        raise RuntimeError(f"Independent recovery verification failed: {verification.get('problems')}")
    return {"result": result, "verification": verification}


def execute_recovery_then_fit67() -> dict[str, Any]:
    recovered = recover_cutoff55()
    cutoff67 = run_cutoff67_with_safe_serializer()
    return {"recovered_cutoff55": recovered, "fit_cutoff67": cutoff67}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"recover55", "fit67", "finalize", "execute"}:
        raise SystemExit("usage: python -m scripts.optimization_v3_classification_sparse_mlp_v1_recovery {recover55|fit67|finalize|execute}")
    command = sys.argv[1]
    if command == "recover55":
        output = recover_cutoff55()
    elif command == "fit67":
        output = run_cutoff67_with_safe_serializer()
    elif command == "finalize":
        output = finalize_and_verify()
    else:
        output = execute_recovery_then_fit67()
    print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
