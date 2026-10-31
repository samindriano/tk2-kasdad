"""Amendment-bound, deterministic replay verification for the fixed two-head pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_evaluation import array_digest, digest_json
from scripts import optimization_v3_cls_residual as run

VERIFY_FREEZE_PATH = run.REPORT_DIR / "verification_freeze_v3_amendment_01.json"
HEAD_CONTEXT_AMENDMENT = run.REPORT_DIR / "verification_amendment_02.json"
FAILED_V1_LOG = run.RUN_DIR / "verify.log"
FAILED_V2_LOG = run.RUN_DIR / "verify_v2.log"


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    """Mirror the frozen pilot metric schema, including Brier and confusion counts."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Metric input is not an aligned finite probability vector")
    pred = p >= 0.5
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos)); tn = int(np.count_nonzero(~pred & ~pos))
    fp = int(np.count_nonzero(pred & ~pos)); fn = int(np.count_nonzero(~pred & pos))
    ppos = tp / (tp + fp) if tp + fp else 0.0
    rpos = tp / (tp + fn) if tp + fn else 0.0
    pneg = tn / (tn + fn) if tn + fn else 0.0
    rneg = tn / (tn + fp) if tn + fp else 0.0
    fpos = 2 * ppos * rpos / (ppos + rpos) if ppos + rpos else 0.0
    fneg = 2 * pneg * rneg / (pneg + rneg) if pneg + rneg else 0.0
    return {"n": int(y.size), "macro_f1": float((fpos + fneg) / 2),
            "f1_positive": float(fpos), "f1_negative": float(fneg),
            "positive_prevalence": float(pos.mean()),
            "positive_prediction_rate": float(pred.mean()),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "error_count": fp + fn,
            "brier": float(np.mean((p - y) ** 2)),
            "threshold": 0.5}


def compare_metric(actual: dict, expected: dict) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"Metric schema mismatch: actual={sorted(actual)} expected={sorted(expected)}")
    for name, value in expected.items():
        if isinstance(value, float):
            if not np.isclose(actual[name], value, rtol=0, atol=1e-12):
                raise AssertionError(f"Metric {name} changed: {actual[name]} != {value}")
        elif actual[name] != value:
            raise AssertionError(f"Metric {name} changed: {actual[name]} != {value}")


def dry_run_metric_schema() -> dict:
    """No-fit smoke check that all result fields, including brier, compare."""
    yy = np.array([0, 1, 0, 1], dtype=np.int8)
    pp = np.array([0.2, 0.8, 0.7, 0.3], dtype=float)
    value = metric(yy, pp)
    compare_metric(value, dict(value))
    expected_schema = {"n", "macro_f1", "f1_positive", "f1_negative", "positive_prevalence",
                       "positive_prediction_rate", "tp", "tn", "fp", "fn", "error_count", "brier", "threshold"}
    if set(value) != expected_schema or not np.isfinite(value["brier"]):
        raise AssertionError("Metric schema smoke check failed")
    report = run.read_json(run.REPORT_DIR / "results.json")
    report_metric = report["evaluation"]["W1"]["A_base_structured"]["present_current"]
    if set(report_metric) != set(value):
        raise AssertionError("Metric fields differ from pilot report schema")
    compare_metric(value, dict(value))
    return {"status": "PASS", "rows": len(yy), "fields": sorted(value),
            "report_schema_fields": sorted(report_metric), "brier": value["brier"]}


def verification_identity() -> dict:
    if not FAILED_V1_LOG.is_file() or not FAILED_V2_LOG.is_file():
        raise FileNotFoundError("Preserved V1/V2 verifier attempt logs are required")
    original_freeze = run.read_json(run.FREEZE_PATH)
    current_fit_identity = run.identity()
    original_fit_identity = original_freeze["identity"]
    old_without_head = dict(original_fit_identity)
    current_without_head = dict(current_fit_identity)
    old_head = old_without_head.pop("source_commit_context")
    current_without_head.pop("source_commit_context")
    if current_without_head != old_without_head:
        raise RuntimeError("A source/protocol/Train/OOF/cache/runtime identity changed beyond HEAD context")
    if not HEAD_CONTEXT_AMENDMENT.is_file():
        raise FileNotFoundError("HEAD-context-only verification amendment is required")
    head_amendment = run.read_json(HEAD_CONTEXT_AMENDMENT)
    if (head_amendment.get("parent_fit_freeze_signature") != original_freeze.get("signature")
            or head_amendment.get("original_source_commit_context") != old_head
            or not isinstance(head_amendment.get("current_source_commit_context"), str)
            or head_amendment.get("identity_diff") != "Only source_commit_context differs. All source, protocol, official Train, prediction-cache, raw OOF input, code and runtime hashes remain identical."):
        raise RuntimeError("HEAD-context amendment does not reconcile the preserved fit freeze")
    report_path = run.REPORT_DIR / "results.json"
    report = run.read_json(report_path)
    artifacts = {}
    for key, item in report["head_metadata"].items():
        model = run.ROOT / item["model_path"]
        prediction = run.ROOT / item["prediction_path"]
        artifacts[key] = {"model_path": item["model_path"], "model_sha256": run.file_sha256(model),
                          "prediction_path": item["prediction_path"], "prediction_sha256": run.file_sha256(prediction)}
    source_files = ["scripts/optimization_v3_cls_residual.py",
                    "scripts/optimization_v3_verify_cls_residual.py",
                    "scripts/optimization_v3_verify_cls_residual_v2.py",
                    "scripts/optimization_v3_verify_cls_residual_v3.py",
                    "src/models/optimization_v3_residual.py"]
    return {
        "parent_fit_freeze_signature": original_freeze["signature"],
        "original_fit_identity_without_head_context_sha256": digest_json(old_without_head),
        "current_fit_identity_without_head_context_sha256": digest_json(current_without_head),
        "original_source_commit_context": old_head,
        "head_context_amendment_sha256": run.file_sha256(HEAD_CONTEXT_AMENDMENT),
        "source_sha256": {path: run.file_sha256(run.ROOT / path) for path in source_files},
        "official_train_sha256": original_freeze["identity"]["official_train_sha256"],
        "oof_and_cache_input_hashes": original_freeze["identity"]["source_cache_sha256"],
        "oof_raw_input_identities": original_freeze["identity"]["oof_training_input_identities"],
        "original_abort_receipt": {"v1_log_sha256": run.file_sha256(FAILED_V1_LOG),
                                   "v1_exit_code": -1073740791,
                                   "v2_log_sha256": run.file_sha256(FAILED_V2_LOG),
                                   "v2_failure_stage": "final metric comparison missing Brier field after the four-head replay loop"},
        "pilot_result_sha256": run.file_sha256(report_path),
        "pilot_artifacts_sha256": artifacts,
        "metric_smoke_check": dry_run_metric_schema(),
        "test_read": False,
        "new_candidate_configuration_or_outer_score": False,
    }


def freeze_verification() -> dict:
    identity = verification_identity()
    signature = digest_json(identity)
    if VERIFY_FREEZE_PATH.exists():
        receipt = run.read_json(VERIFY_FREEZE_PATH)
        if digest_json(receipt.get("identity")) != signature or receipt.get("signature") != signature:
            raise RuntimeError("Existing v3 verification freeze differs; preserve it and create a new verifier version")
        return receipt
    receipt = {"status": "FROZEN_BEFORE_V3_DETERMINISTIC_VERIFICATION_REPLAY",
               "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "identity": identity, "signature": signature,
               "source_boundary": "Deterministic replay of the already frozen four pilot fits; inputs include original abort receipt and result/model/prediction hashes. No Test or Kaggle access."}
    run.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_FREEZE_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def check_verification_freeze() -> dict:
    if not VERIFY_FREEZE_PATH.is_file():
        raise RuntimeError("Freeze v3 verifier/source/results/inputs before deterministic replay")
    receipt = run.read_json(VERIFY_FREEZE_PATH)
    current = verification_identity()
    if digest_json(current) != receipt.get("signature") or digest_json(receipt.get("identity")) != receipt.get("signature"):
        raise RuntimeError("V3 verifier source, results, caches or frozen inputs changed after verification freeze")
    return receipt


def independent_views(y, cycle, verified_vectors, expected_report):
    sections = {"W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
                "H12_E": [(55, 56, 67)], "H12_L": [(67, 68, 79)]}
    sections["W1_plus_W4_pilot_slice"] = sections["W1"] + sections["W4"]
    sections["H12_pooled"] = sections["H12_E"] + sections["H12_L"]
    output = {}
    for view, parts in sections.items():
        output[view] = {}
        for head_name in run.HEADS:
            ys, current, candidate, missing = [], [], [], []
            for cutoff, lo, hi in parts:
                rec = verified_vectors[cutoff][head_name]
                rows = rec["rows"]
                keep = (cycle[rows] >= lo) & (cycle[rows] <= hi)
                ys.append(y[rows][keep]); current.append(rec["champion"][keep])
                candidate.append(rec["candidate"][keep]); missing.append(rec["missing"][keep])
            yy = np.concatenate(ys); p0 = np.concatenate(current); p1 = np.concatenate(candidate)
            miss = np.concatenate(missing).astype(bool); present = ~miss
            observed = {"all_current": metric(yy, p0), "all_candidate": metric(yy, p1),
                        "present_current": metric(yy[present], p0[present]),
                        "present_candidate": metric(yy[present], p1[present]),
                        "missing_current": metric(yy[miss], p0[miss]),
                        "missing_candidate": metric(yy[miss], p1[miss]),
                        "missing_exact_match": bool(np.array_equal(p1[miss], p0[miss]))}
            observed["all_delta_macro_f1"] = observed["all_candidate"]["macro_f1"] - observed["all_current"]["macro_f1"]
            observed["present_delta_macro_f1"] = observed["present_candidate"]["macro_f1"] - observed["present_current"]["macro_f1"]
            for key, value in observed.items():
                expected = expected_report[view][head_name][key]
                if isinstance(value, dict):
                    compare_metric(value, expected)
                elif isinstance(value, float):
                    if not np.isclose(value, expected, rtol=0, atol=1e-12):
                        raise AssertionError(f"{view}/{head_name}/{key} differs")
                elif value != expected:
                    raise AssertionError(f"{view}/{head_name}/{key} differs")
            output[view][head_name] = observed
    return output


def verify() -> dict:
    verify_freeze = check_verification_freeze()
    # Verify every fit-source/data hash while treating Git HEAD strictly as
    # contextual because the repository advanced after the committed freeze.
    original_fit_freeze = run.read_json(run.FREEZE_PATH)
    current_fit_identity = run.identity()
    old_identity = dict(original_fit_freeze["identity"])
    new_identity = dict(current_fit_identity)
    old_identity.pop("source_commit_context")
    new_identity.pop("source_commit_context")
    if new_identity != old_identity:
        raise AssertionError("Fit source/data identity changed beyond contextual HEAD")
    fit_freeze = original_fit_freeze
    report_path = run.REPORT_DIR / "results.json"
    report = run.read_json(report_path)
    if report["freeze_signature"] != fit_freeze["signature"]:
        raise AssertionError("Pilot output differs from original pre-fit freeze")
    frame, y, cycle, missing, _ = run.load_train()
    vectors, head_checks = {}, {}
    for cutoff in run.OUTER:
        outer = run.outer_inputs(frame, y, cycle, missing, cutoff)
        vectors[cutoff] = {}
        for head_name, score_names in run.HEADS.items():
            key = f"{cutoff}/{head_name}"
            meta = report["head_metadata"][key]
            oof = run.prepare_oof(frame, y, cycle, missing, cutoff, head_name)
            blocks = oof["blocks"]
            if [b["fit_cutoff"] for b in blocks] != list(range(51, cutoff, 4)):
                raise AssertionError(f"OOF cutoff sequence differs for {key}")
            block_rows = np.concatenate([b["rows"] for b in blocks])
            if len(np.unique(block_rows)) != len(block_rows):
                raise AssertionError(f"OOF blocks overlap for {key}")
            for block in blocks:
                lo = block["fit_cutoff"] + 1; hi = block["fit_cutoff"] + 4
                if hi > cutoff or not np.array_equal(block["rows"], run.expected_rows(cycle, lo, hi)):
                    raise AssertionError(f"OOF block is not the next exact four cycles for {key}")
            if cycle[oof["present_rows"]].min() < 52 or cycle[oof["present_rows"]].max() != cutoff:
                raise AssertionError(f"OOF head training rows cross cutoff for {key}")
            if meta["oof_raw_input_sha256"] != oof["raw_input_sha256"]:
                raise AssertionError(f"Frozen raw OOF input hash differs for {key}")
            if meta["state"]["rows"] != len(oof["present_rows"]):
                raise AssertionError(f"OOF meta training denominator differs for {key}")
            pred_path = run.ROOT / meta["prediction_path"]
            model_path = run.ROOT / meta["model_path"]
            if run.file_sha256(pred_path) != meta["prediction_sha256"] or run.file_sha256(model_path) != meta["model_sha256"]:
                raise AssertionError(f"Frozen model/prediction artifact hash differs for {key}")
            with np.load(pred_path, allow_pickle=False) as z:
                saved = {name: z[name].copy() for name in z.files}
            rows = outer["rows"]; miss = missing[rows]
            if not np.array_equal(saved["outer_rows"], rows) or not np.array_equal(saved["missing_mask"].astype(bool), miss):
                raise AssertionError(f"Outer row IDs or stream mask differs for {key}")
            for name, value in (("base_probability", outer["base"]),
                                ("cb_probability", outer["cb"]),
                                ("champion_probability", outer["champion"])):
                if not np.array_equal(saved[name], value):
                    raise AssertionError(f"Current control differs for {key}/{name}")
            if not np.array_equal(saved["oof_present_rows"], oof["present_rows"]):
                raise AssertionError(f"Saved meta-training row IDs differ for {key}")
            if not np.array_equal(saved["oof_scores"], oof["scores"]) or not np.array_equal(saved["oof_target"], oof["y"]):
                raise AssertionError(f"Saved OOF score/target vectors differ for {key}")
            head = OOFResidualHead("classification", score_names)
            with threadpool_limits(limits=2):
                head.fit(oof["X"], oof["scores"], oof["y"])
            state = head.audit_state()
            for field in ("fit_data_sha256", "fit_scores_sha256", "target_sha256", "booster_sha256"):
                if state[field] != meta["state"][field]:
                    raise AssertionError(f"Deterministic head state differs for {key}/{field}")
            if state["rows"] != meta["state"]["rows"] or state["preprocessor"] != meta["state"]["preprocessor"]:
                raise AssertionError(f"Fit-only preprocessing state differs for {key}")
            model_text = head.model.booster_.model_to_string()
            if hashlib.sha256(model_text.encode("utf-8")).hexdigest() != meta["state"]["booster_sha256"] or model_text != model_path.read_text(encoding="utf-8"):
                raise AssertionError(f"Replayed LightGBM model text differs for {key}")
            present = ~miss
            X_outer = frame.iloc[rows[present]][list(run.EXPANDED_FEATURES)].copy()
            score_idx = [0] if head_name == "A_base_structured" else [0, 1]
            outer_scores = np.column_stack([outer["base"], outer["cb"]])[present][:, score_idx]
            design, _, _ = head._design(X_outer, outer_scores, False)
            design_array = design.to_numpy(dtype=float)
            if not np.array_equal(design_array, saved["outer_design_present"]) or array_digest(design_array) != meta["outer_design_sha256"]:
                raise AssertionError(f"Outer transform differs for {key}")
            margin = head.model.booster_.predict(design, raw_score=True, num_threads=2)
            if not np.array_equal(margin, saved["raw_correction_present"]) or array_digest(margin) != meta["raw_correction_sha256"]:
                raise AssertionError(f"Raw margin differs for {key}")
            corrected = expit(logit(np.clip(outer["base"][present], 1e-6, 1 - 1e-6)) + 0.5 * margin)
            candidate = outer["champion"].copy(); candidate[present] = corrected
            if not np.array_equal(candidate, saved["candidate_probability"]) or not np.array_equal(candidate[miss], outer["champion"][miss]):
                raise AssertionError(f"Base-margin formula or exact missing route differs for {key}")
            vectors[cutoff][head_name] = {"rows": rows, "champion": outer["champion"],
                                          "candidate": candidate, "missing": miss}
            head_checks[key] = {"oof_present_rows": int(len(oof["present_rows"])),
                                "outer_present_rows": int(present.sum()),
                                "outer_missing_rows": int(miss.sum()),
                                "raw_margin_sha256": array_digest(margin),
                                "candidate_present_sha256": array_digest(candidate[present]),
                                "missing_route_exact_match": True}
    evaluation = independent_views(y, cycle, vectors, report["evaluation"])
    individual = [evaluation[v][h]["present_delta_macro_f1"]
                  for v in ("W1", "W4", "H12_E", "H12_L") for h in run.HEADS]
    pooled_gain = any(evaluation[v][h]["present_delta_macro_f1"] > 0
                      for v in ("W1_plus_W4_pilot_slice", "H12_pooled") for h in run.HEADS)
    gate = all(value >= -0.01 for value in individual) and pooled_gain
    if report["gate"]["overall_pass"] is not gate:
        raise AssertionError("Continuation gate differs from independent verification")
    result = {"status": "PASS", "verification_version": 3,
              "verification_freeze_signature": verify_freeze["signature"],
              "parent_fit_freeze_signature": fit_freeze["signature"],
              "amendment_bound_deterministic_replay_fits": 4,
              "previous_failed_attempt_replays_not_accepted_as_verification": 5,
              "candidate_configurations_or_outer_scores_added": 0,
              "test_read": False, "kaggle_read_or_submit": False,
              "outer_labels_used_for_fit_or_selection": False,
              "checks": {"OOF cutoff/block/nonduplication": True,
                         "Train and raw score input hashes": True,
                         "fit state/preprocessor/model fingerprints": True,
                         "raw margin bitwise reconstruction": True,
                         "base_logit_plus_fixed_shrinkage bitwise reconstruction": True,
                         "missing route exact preservation": True,
                         "metric schema including brier": True,
                         "outer metrics and continuation gate": True},
              "head_checks": head_checks, "evaluation_views": evaluation,
              "gate_pass": gate}
    out = run.REPORT_DIR / "verification_v3.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    amendment = {"status": "VERIFIER_V3_AMENDMENT_BOUND_BEFORE_REPLAY",
                 "parent_fit_freeze_signature": fit_freeze["signature"],
                 "verification_freeze_signature": verify_freeze["signature"],
                 "preserved_v1_sha256": verify_freeze["identity"]["source_sha256"]["scripts/optimization_v3_verify_cls_residual.py"],
                 "preserved_v2_sha256": verify_freeze["identity"]["source_sha256"]["scripts/optimization_v3_verify_cls_residual_v2.py"],
                 "v3_verifier_sha256": verify_freeze["identity"]["source_sha256"]["scripts/optimization_v3_verify_cls_residual_v3.py"],
                 "original_failure_receipts": verify_freeze["identity"]["original_abort_receipt"],
                 "pilot_result_sha256": verify_freeze["identity"]["pilot_result_sha256"],
                 "pilot_artifacts_sha256": verify_freeze["identity"]["pilot_artifacts_sha256"],
                 "verification_result_sha256": run.file_sha256(out),
                 "candidate_fit_outputs_changed": False,
                 "accepted_verification_replay_fits": 4,
                 "prior_failed_verifier_replay_attempts": 5,
                 "new_candidate_configurations_or_outer_scores": 0,
                 "preserved": ["original fit freezes", "V1 and V2 verifier sources", "pilot report/results", "model/prediction caches"]}
    amendment_path = run.REPORT_DIR / "verification_amendment_01.json"
    if amendment_path.exists():
        raise FileExistsError(f"Refusing to overwrite {amendment_path}")
    amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.verify:
        parser.error("Choose exactly one: --freeze-only or --verify")
    if args.freeze_only:
        smoke = dry_run_metric_schema()
        receipt = freeze_verification()
        print(json.dumps({"status": receipt["status"], "signature": receipt["signature"],
                          "metric_schema_smoke": smoke}, indent=2))
    else:
        print(json.dumps(verify(), indent=2, sort_keys=True))
