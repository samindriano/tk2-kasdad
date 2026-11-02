"""Independent verifier for MLP outputs recovered after a post-fit serializer error."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from scripts import optimization_v3_classification_sparse_mlp_v1 as runner
from scripts import optimization_v3_classification_sparse_mlp_v1_verify as independent
from scripts.research_campaign_classification import load_train
from src.data.audit import sha256 as file_sha256
from src.models.classification_sparse_mlp_v1 import transform_score_design
from src.models.research_next_budget import ledger_summary, read_events
from src.models.research_next_contract import read_bundle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/optimization_v3/classification_mlp_v1"
RUNS = ROOT / "experiments/runs/optimization_v3_classification_mlp_v1"
FREEZE = ROOT / "experiments/optimization_v3/classification_mlp_v1/freeze.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RECOVERY = ROOT / "scripts/optimization_v3_classification_sparse_mlp_v1_recovery.py"


def _sha(path: Path) -> str:
    return file_sha256(path)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return bool(np.isclose(float(a), float(b), rtol=0, atol=1e-12))
    return a == b


def _recovery_event_hashes_match(
    event: dict[str, Any], *, model_sha: str, state_sha: str,
    receipt_sha: str, recovery_sha: str,
) -> bool:
    return (
        event.get("model_joblib_sha256") == model_sha and
        event.get("fit_state_sha256") == state_sha and
        event.get("bundle_receipt_sha256") == receipt_sha and
        event.get("recovery_code_sha256") == recovery_sha
    )


def verify_recovered() -> dict[str, Any]:
    problems: list[str] = []
    frozen = _json(FREEZE)
    identity = frozen["identity"]
    signature = frozen["signature"]
    if _sha(RECOVERY) == "":
        problems.append("recovery source sha is unavailable")
    if not independent._identity_audit()[2] == []:
        problems.extend([f"frozen identity: {item}" for item in independent._identity_audit()[2]])

    frame, target, train_path = load_train(None)
    if len(frame) != 50_608 or _sha(train_path) != identity["input_sha256"].get("data/raw/classification/Train.csv"):
        problems.append("official Train identity mismatch")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80)):
        problems.append("official Train cycles differ from 47-79")
    baselines = {cutoff: independent._reconstruct_baseline(frame, target, cutoff) for cutoff in runner.ORIGINS}
    events = read_events(LEDGER)
    outputs: dict[int, dict[str, Any]] = {}
    fit_audits: dict[str, Any] = {}
    recovery_chain_valid = False
    cutoff67_saved_valid = False

    for cutoff in runner.ORIGINS:
        fit_id = f"{runner.CARD_ID}_{cutoff}"
        starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
        terminals = [event for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and event.get("fit_id") == fit_id]
        recoveries = [event for event in events if event.get("event") == "FIT_OUTPUT_RECOVERED" and event.get("fit_id") == fit_id]
        if len(starts) != 1 or len(terminals) != 1:
            problems.append(f"ledger start/terminal cardinality mismatch at cutoff {cutoff}")
            continue
        if (starts[0].get("card_id") != runner.CARD_ID or starts[0].get("stage") != runner.STAGE or
                starts[0].get("cutoff") != cutoff or starts[0].get("fit_consumed_at_start") is not True):
            problems.append(f"ledger FIT_STARTED identity invalid at cutoff {cutoff}")
        if cutoff == 55:
            if terminals[0].get("event") != "FIT_FAILED" or len(recoveries) != 1:
                problems.append("cutoff55 original failure and artifact recovery chain not preserved")
            elif (terminals[0].get("card_id") != runner.CARD_ID or terminals[0].get("stage") != runner.STAGE or
                  terminals[0].get("cutoff") != 55 or
                  terminals[0].get("error") != "TypeError: Object of type int64 is not JSON serializable"):
                problems.append("cutoff55 FIT_FAILED identity/error differs from preserved technical failure")
            elif (recoveries[0].get("card_id") != runner.CARD_ID or recoveries[0].get("stage") != runner.STAGE or
                  recoveries[0].get("cutoff") != 55 or recoveries[0].get("original_terminal_status") != "FIT_FAILED" or
                  recoveries[0].get("original_terminal_event_preserved") is not True or
                  recoveries[0].get("refit_performed") is not False):
                problems.append("cutoff55 recovery ledger identity/metadata invalid")
        elif terminals[0].get("event") != "FIT_SAVED" or recoveries:
            problems.append("cutoff67 must be one normal saved fit with no recovery event")
        elif (terminals[0].get("card_id") != runner.CARD_ID or terminals[0].get("stage") != runner.STAGE or
              terminals[0].get("cutoff") != 67):
            problems.append("cutoff67 FIT_SAVED ledger identity invalid")

        fit_dir = RUNS / f"fit{cutoff}"
        model_path = fit_dir / "fitted_state.joblib"
        state_path = fit_dir / "fit_state.json"
        if not model_path.is_file() or not state_path.is_file():
            problems.append(f"durable fitted state missing at cutoff {cutoff}")
            continue
        model_sha = _sha(model_path)
        state = _json(state_path)
        if state.get("model_joblib_sha256") != model_sha or state.get("freeze_signature") != signature:
            problems.append(f"model/state identity mismatch at cutoff {cutoff}")
        if state.get("input_hashes") != identity["input_sha256"]:
            problems.append(f"state input hashes mismatch at cutoff {cutoff}")
        historical_state = _json(ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json")["C2_B"]["fitted_state"]
        pre_state = state.get("preprocessing_state", {})
        if pre_state.get("structured_preprocessor") != historical_state.get("preprocessor"):
            problems.append(f"saved structured preprocessing audit differs from historical C2_B at cutoff {cutoff}")
        if pre_state.get("tfidf_idf_sha256") != historical_state.get("text", {}).get("idf_sha256"):
            problems.append(f"saved TF-IDF IDF digest differs from historical C2_B at cutoff {cutoff}")
        parity = pre_state.get("c2b_historical_preprocessing_parity")
        if not isinstance(parity, dict) or parity.get("status") != "PASS" or not all(parity.get("checks", {}).values()):
            problems.append(f"saved C2_B preprocessing parity receipt is missing or failed at cutoff {cutoff}")

        stored = joblib.load(model_path)
        base, estimator = stored["base"], stored["estimator"]
        expected_fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
        expected_score_rows = baselines[cutoff]["rows"]
        feature_names = list(base.input_feature_names)
        from src.models.round1_models import _fit_fingerprint
        X_fit = frame.iloc[expected_fit_rows].loc[:, feature_names]
        X_score = frame.iloc[expected_score_rows].loc[:, feature_names]
        y_fit = np.asarray(target)[expected_fit_rows]
        if base._fit_fingerprint != _fit_fingerprint(X_fit, tuple(feature_names), y_fit):
            problems.append(f"fit source fingerprint mismatch at cutoff {cutoff}")
        if base._fit_fingerprint != _json(ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json")["C2_B"]["fitted_state"]["fit_data_sha256"]:
            problems.append(f"preprocessing does not use exact C2_B fit population at cutoff {cutoff}")
        expected_params = _json(ROOT / "experiments/optimization_v3/classification_mlp_v1/protocol.json")["model"]["parameters"]
        actual_params = estimator.get_params(deep=True)
        actual_params = {key: list(value) if isinstance(value, tuple) else value for key, value in actual_params.items()}
        if set(expected_params) != set(actual_params) or any(actual_params[key] != expected_params[key] for key in expected_params):
            problems.append(f"fitted MLP params differ from frozen protocol at cutoff {cutoff}")
        if int(estimator.n_iter_) != 30 or len(estimator.loss_curve_) != 30 or not np.isfinite(estimator.loss_curve_).all():
            problems.append(f"fit schedule/loss curve invalid at cutoff {cutoff}")

        score_matrix = transform_score_design(base, X_score)
        raw = np.asarray(estimator.predict_proba(score_matrix)[:, 1], dtype=np.float64)
        if np.asarray(estimator.classes_).tolist() != [0, 1]:
            problems.append(f"positive class order differs at cutoff {cutoff}")
        baseline = baselines[cutoff]
        candidate = np.where(baseline["missing"], baseline["cb"], .75 * baseline["baseline"] + .25 * raw)
        matched = np.where(baseline["missing"], baseline["cb"], .75 * baseline["baseline"] + .25 * baseline["c2b"])
        diagnostic = np.where(baseline["missing"], baseline["cb"], baseline["c2b"])
        ids = frame.iloc[expected_score_rows]["record_id"].astype(str).to_numpy()
        bundle_identity = runner._bundle_identity(identity, signature, cutoff, model_sha, _sha(state_path))
        try:
            bundle = read_bundle(fit_dir / "bundle", bundle_identity, task="classification")
            arrays = bundle.arrays
        except Exception as exc:
            problems.append(f"bundle contract failed at cutoff {cutoff}: {type(exc).__name__}: {exc}")
            continue
        for field, expected in (("score_rows", expected_score_rows), ("score_record_ids", ids),
                                ("cycle_code", cycles[expected_score_rows])):
            if not np.array_equal(arrays[field], expected):
                problems.append(f"bundle {field} mismatch at cutoff {cutoff}")
        checks = {
            "raw_mlp_rebuilt": np.array_equal(arrays["raw_mlp_probability"], raw),
            "candidate_route_rebuilt": np.array_equal(arrays["prediction"], candidate),
            "matched_route_rebuilt": np.array_equal(arrays["matched_prediction"], matched),
            "sparse25_baseline_exact": np.array_equal(arrays["baseline_prediction"], baseline["baseline"]),
            "C2_B_exact": np.array_equal(arrays["control_c2b_probability"], baseline["c2b"]),
            "CB_exact": np.array_equal(arrays["control_cb_probability"], baseline["cb"]),
            "missing_mask_exact": np.array_equal(arrays["control_stream_missing"].astype(bool), baseline["missing"]),
        }
        if not all(checks.values()):
            problems.append(f"bundle route/control reconstruction failed at cutoff {cutoff}: {checks}")
        if cutoff == 55 and recoveries and recoveries[0].get("bundle_receipt_sha256") != bundle.receipt["receipt_sha256"]:
            problems.append("cutoff55 recovery event receipt differs from recovered bundle")
        if cutoff == 55 and recoveries and recoveries[0].get("recovery_code_sha256") != _sha(RECOVERY):
            problems.append("cutoff55 recovery event source hash differs from verified recovery code")
        if cutoff == 55 and recoveries:
            recovery = recoveries[0]
            exact_recovery_hashes = _recovery_event_hashes_match(
                recovery, model_sha=model_sha, state_sha=_sha(state_path),
                receipt_sha=bundle.receipt["receipt_sha256"], recovery_sha=_sha(RECOVERY),
            )
            if not exact_recovery_hashes:
                problems.append("cutoff55 recovery event artifact hashes do not match model/state/bundle/code")
            if terminals[0].get("error") != "TypeError: Object of type int64 is not JSON serializable":
                problems.append("cutoff55 preserved FIT_FAILED is not the authorized post-fit serializer failure")
            recovery_chain_valid = bool(
                exact_recovery_hashes and recovery.get("card_id") == runner.CARD_ID and
                recovery.get("stage") == runner.STAGE and recovery.get("cutoff") == 55 and
                recovery.get("original_terminal_status") == "FIT_FAILED" and
                recovery.get("original_terminal_event_preserved") is True and
                recovery.get("refit_performed") is False and terminals[0].get("event") == "FIT_FAILED" and
                terminals[0].get("fit_id") == f"{runner.CARD_ID}_55" and
                terminals[0].get("error") == "TypeError: Object of type int64 is not JSON serializable"
            )
        if cutoff == 67 and terminals[0].get("receipt_sha256") != bundle.receipt["receipt_sha256"]:
            problems.append("cutoff67 FIT_SAVED receipt differs from bundle")
        if cutoff == 67:
            cutoff67_saved_valid = bool(
                terminals[0].get("event") == "FIT_SAVED" and
                terminals[0].get("receipt_sha256") == bundle.receipt["receipt_sha256"] and
                terminals[0].get("card_id") == runner.CARD_ID and
                terminals[0].get("stage") == runner.STAGE and terminals[0].get("cutoff") == 67
            )
        outputs[cutoff] = {
            "score_rows": arrays["score_rows"].astype(np.int64),
            "score_record_ids": arrays["score_record_ids"].astype(str),
            "cycle_code": cycles[expected_score_rows], "prediction": candidate,
            "baseline_prediction": baseline["baseline"], "matched_prediction": matched,
            "c2b_diagnostic": diagnostic, "origin": np.full(len(expected_score_rows), cutoff, dtype=np.int64),
        }
        fit_audits[str(cutoff)] = {"fit_rows": len(expected_fit_rows), "score_rows": len(expected_score_rows),
            "n_iter": int(estimator.n_iter_), "loss_curve": [float(v) for v in estimator.loss_curve_],
            "model_joblib_sha256": model_sha, "fit_state_sha256": _sha(state_path),
            "bundle_receipt_sha256": bundle.receipt["receipt_sha256"], "route_checks": checks,
            "ledger_terminal": terminals[0]["event"],
            "recovered_after_failure": bool(cutoff == 55),
            "refit_performed_during_recovery": False if cutoff == 55 else None}

    recomputed_metrics = {}
    recomputed_groups = []
    recomputed_cycles = []
    recomputed_gate = {}
    if len(outputs) == len(runner.ORIGINS):
        recomputed_metrics, recomputed_groups, recomputed_cycles, recomputed_gate = independent._views(outputs, frame, target)
        report_path = REPORT / "results.json"
        if not report_path.is_file():
            problems.append("results.json missing")
        else:
            report = _json(report_path)
            if not _same(report.get("metrics_by_view"), recomputed_metrics):
                problems.append("independent view metrics differ from saved report")
            if not _same(report.get("protected_groups"), recomputed_groups):
                problems.append("independent subgroup metrics differ from saved report")
            if not _same(report.get("per_cycle"), recomputed_cycles):
                problems.append("independent cycle metrics differ from saved report")
            if not _same(report.get("frozen_gate"), recomputed_gate):
                problems.append("independent frozen gate differs from saved report")
            if report.get("scientific_verdict") != recomputed_gate.get("status"):
                problems.append("scientific verdict label mismatch")
            if report.get("status") != "COMPLETE_REVIEW_REQUIRED":
                problems.append("report checkpoint status is not complete review required")

    budget = ledger_summary(LEDGER)
    if budget["started_and_consumed"] != 50 or budget["saved"] != 48 or budget["failed_or_interrupted_started"] != 2:
        problems.append("budget totals differ from expected two MLP fits plus preserved prior history")
    result = {
        "status": "PASS" if not problems else "FAIL",
        "campaign": "CLASSIFICATION_SPARSE_MLP_V1_RECOVERY_VERIFICATION",
        "freeze_signature": signature, "recovery_code_sha256": _sha(RECOVERY),
        "checks": {
            "original_frozen_sources_and_inputs": not independent._identity_audit()[2],
            "official_train_rows_hash_and_cycles": len(frame) == 50_608 and _sha(train_path) == identity["input_sha256"].get("data/raw/classification/Train.csv") and np.array_equal(np.unique(cycles), np.arange(47, 80)),
            "historical_control_routes_rebuilt_independently": len(baselines) == 2,
            "both_model_states_and_routes_recomputed": len(outputs) == 2,
            "metrics_groups_cycles_and_gate_recomputed": len(recomputed_metrics) == 6,
            "cutoff55_failed_event_preserved_and_recovered_without_refit": recovery_chain_valid and sum(e.get("event")=="FIT_OUTPUT_RECOVERED" and e.get("fit_id")==f"{runner.CARD_ID}_55" for e in events) == 1,
            "cutoff67_one_saved_fit": cutoff67_saved_valid and sum(e.get("event")=="FIT_SAVED" and e.get("fit_id")==f"{runner.CARD_ID}_67" for e in events) == 1,
            "test_sample_submission_and_kaggle_untouched": identity.get("test_read") is False and identity.get("sample_submission_read") is False and identity.get("kaggle_access") is False,
        },
        "fit_audits": fit_audits,
        "independent_metrics_by_view": recomputed_metrics,
        "independent_protected_groups": recomputed_groups,
        "independent_per_cycle": recomputed_cycles,
        "independent_frozen_gate": recomputed_gate,
        "budget": budget,
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
        "problems": problems,
    }
    path = REPORT / "independent_verification.json"
    encoded = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"Refusing to overwrite different independent verification: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify_recovered(), ensure_ascii=False, allow_nan=False, indent=2))
