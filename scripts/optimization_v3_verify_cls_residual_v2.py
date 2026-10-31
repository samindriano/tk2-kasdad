"""Versioned deterministic replay verifier for the V3 classification pilot.

The first verifier attempted LightGBM's C++ model-file loader and aborted on
this runtime's serialized model text. This verifier preserves that source and
instead refits each fixed head from the frozen OOF rows, then reconstructs its
raw margin and base-logit addition directly from the deterministic booster.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_evaluation import array_digest
from scripts import optimization_v3_cls_residual as run


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Score vector is not aligned finite probability output")
    pred = p >= 0.5
    return {"n": int(y.size),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "error_count": int(np.count_nonzero(pred != y))}


def compare_metric(observed: dict, expected: dict) -> None:
    for name, value in expected.items():
        if isinstance(value, float):
            if not np.isclose(observed[name], value, rtol=0, atol=1e-12):
                raise AssertionError(f"Metric {name} changed: {observed[name]} vs {value}")
        elif observed[name] != value:
            raise AssertionError(f"Metric {name} changed: {observed[name]} vs {value}")


def independent_views(frame, y, cycle, verified_vectors, expected_report):
    sections = {"W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
                "H12_E": [(55, 56, 67)], "H12_L": [(67, 68, 79)]}
    sections["W1_plus_W4_pilot_slice"] = sections["W1"] + sections["W4"]
    sections["H12_pooled"] = sections["H12_E"] + sections["H12_L"]
    out = {}
    for view, parts in sections.items():
        out[view] = {}
        for head in run.HEADS:
            Ys, current, candidate, miss = [], [], [], []
            for cutoff, lo, hi in parts:
                rec = verified_vectors[cutoff][head]
                outer_rows = rec["rows"]
                keep = (cycle[outer_rows] >= lo) & (cycle[outer_rows] <= hi)
                rows = outer_rows[keep]
                Ys.append(y[rows])
                current.append(rec["champion"][keep])
                candidate.append(rec["candidate"][keep])
                miss.append(rec["missing"][keep])
            yy = np.concatenate(Ys)
            base = np.concatenate(current)
            challenger = np.concatenate(candidate)
            is_missing = np.concatenate(miss).astype(bool)
            present = ~is_missing
            observed = {
                "all_current": metric(yy, base), "all_candidate": metric(yy, challenger),
                "present_current": metric(yy[present], base[present]),
                "present_candidate": metric(yy[present], challenger[present]),
                "missing_current": metric(yy[is_missing], base[is_missing]),
                "missing_candidate": metric(yy[is_missing], challenger[is_missing]),
                "missing_exact_match": bool(np.array_equal(challenger[is_missing], base[is_missing])),
            }
            observed["all_delta_macro_f1"] = observed["all_candidate"]["macro_f1"] - observed["all_current"]["macro_f1"]
            observed["present_delta_macro_f1"] = observed["present_candidate"]["macro_f1"] - observed["present_current"]["macro_f1"]
            for field, value in observed.items():
                expected = expected_report[view][head][field]
                if isinstance(value, dict):
                    compare_metric(value, expected)
                elif isinstance(value, float):
                    if not np.isclose(value, expected, rtol=0, atol=1e-12):
                        raise AssertionError(f"{view}/{head}/{field} changed")
                elif value != expected:
                    raise AssertionError(f"{view}/{head}/{field} changed")
            out[view][head] = observed
    return out


def verify() -> dict:
    # The original fit freeze binds the original runner/verifier and every data
    # source. This v2 file is an additive, versioned verifier, never a fit input.
    freeze = run.check_freeze()
    result_path = run.REPORT_DIR / "results.json"
    report = run.read_json(result_path)
    if report.get("freeze_signature") != freeze["signature"]:
        raise AssertionError("Pilot result is detached from the pre-fit freeze")
    frame, y, cycle, missing, _ = run.load_train()
    verified_vectors = {}
    head_checks = {}
    for cutoff in run.OUTER:
        outer = run.outer_inputs(frame, y, cycle, missing, cutoff)
        verified_vectors[cutoff] = {}
        for head_name, score_names in run.HEADS.items():
            key = f"{cutoff}/{head_name}"
            meta = report["head_metadata"][key]
            oof = run.prepare_oof(frame, y, cycle, missing, cutoff, head_name)
            # Independent boundary checks against block declarations in the
            # frozen protocol, including exact non-overlap and fit cutoff < row.
            expected_cutoffs = list(range(51, cutoff, 4))
            if [b["fit_cutoff"] for b in oof["blocks"]] != expected_cutoffs:
                raise AssertionError(f"Unexpected OOF fit cutoffs for {key}")
            block_rows = np.concatenate([b["rows"] for b in oof["blocks"]])
            if len(np.unique(block_rows)) != len(block_rows):
                raise AssertionError(f"Duplicate OOF source rows for {key}")
            for block in oof["blocks"]:
                lo, hi = block["fit_cutoff"] + 1, block["fit_cutoff"] + 4
                if hi > cutoff or not np.array_equal(block["rows"], run.expected_rows(cycle, lo, hi)):
                    raise AssertionError(f"OOF block crosses cutoff or is misaligned for {key}")
            if cycle[oof["present_rows"]].min() < 52 or cycle[oof["present_rows"]].max() != cutoff:
                raise AssertionError(f"Meta training row cycle boundary differs for {key}")
            if meta["oof_raw_input_sha256"] != oof["raw_input_sha256"]:
                raise AssertionError(f"Frozen OOF raw input digest differs for {key}")
            if meta["state"]["rows"] != len(oof["present_rows"]):
                raise AssertionError(f"Head fit support differs for {key}")

            artifact_path = run.ROOT / meta["prediction_path"]
            model_path = run.ROOT / meta["model_path"]
            if run.file_sha256(artifact_path) != meta["prediction_sha256"] or run.file_sha256(model_path) != meta["model_sha256"]:
                raise AssertionError(f"Saved prediction or model text hash differs for {key}")
            with np.load(artifact_path, allow_pickle=False) as z:
                saved = {name: z[name].copy() for name in z.files}
            expected_rows = outer["rows"]
            outer_missing = missing[expected_rows]
            if not np.array_equal(saved["outer_rows"], expected_rows):
                raise AssertionError(f"Outer row IDs differ for {key}")
            if not np.array_equal(saved["missing_mask"].astype(bool), outer_missing):
                raise AssertionError(f"Missing-stream mask differs for {key}")
            for name, value in (("base_probability", outer["base"]),
                                ("cb_probability", outer["cb"]),
                                ("champion_probability", outer["champion"])):
                if not np.array_equal(saved[name], value):
                    raise AssertionError(f"Frozen control vector {name} differs for {key}")
            if not np.array_equal(saved["oof_present_rows"], oof["present_rows"]):
                raise AssertionError(f"Saved OOF row IDs differ for {key}")
            if not np.array_equal(saved["oof_scores"], oof["scores"]) or not np.array_equal(saved["oof_target"], oof["y"]):
                raise AssertionError(f"Saved OOF score/target vectors differ for {key}")

            # Deterministic replay verifies the fit-only state and booster hash.
            replay = OOFResidualHead("classification", score_names)
            with threadpool_limits(limits=2):
                replay.fit(oof["X"], oof["scores"], oof["y"])
            replay_state = replay.audit_state()
            for field in ("fit_data_sha256", "fit_scores_sha256", "target_sha256", "booster_sha256"):
                if replay_state[field] != meta["state"][field]:
                    raise AssertionError(f"Deterministic replay state differs at {key}/{field}")
            if replay_state["preprocessor"] != meta["state"]["preprocessor"] or replay_state["rows"] != meta["state"]["rows"]:
                raise AssertionError(f"Fit-only preprocessor state differs for {key}")
            model_text = replay.model.booster_.model_to_string()
            if hashlib.sha256(model_text.encode("utf-8")).hexdigest() != meta["state"]["booster_sha256"]:
                raise AssertionError(f"Replayed LightGBM model text hash differs for {key}")
            if model_text != model_path.read_text(encoding="utf-8"):
                raise AssertionError(f"Model text differs from deterministic replay for {key}")

            present = ~outer_missing
            X_outer = frame.iloc[expected_rows[present]][list(run.EXPANDED_FEATURES)].copy()
            score_idx = [0] if head_name == "A_base_structured" else [0, 1]
            outer_scores = np.column_stack([outer["base"], outer["cb"]])[present][:, score_idx]
            design, _, _ = replay._design(X_outer, outer_scores, False)
            design_array = design.to_numpy(dtype=float)
            if not np.array_equal(design_array, saved["outer_design_present"]):
                raise AssertionError(f"Outer transformed design differs for {key}")
            if array_digest(design_array) != meta["outer_design_sha256"]:
                raise AssertionError(f"Outer design fingerprint differs for {key}")
            raw_margin = replay.model.booster_.predict(design, raw_score=True, num_threads=2)
            if not np.array_equal(raw_margin, saved["raw_correction_present"]):
                raise AssertionError(f"Raw correction margin is not bit-identical for {key}")
            if array_digest(raw_margin) != meta["raw_correction_sha256"]:
                raise AssertionError(f"Raw correction hash differs for {key}")
            correction = expit(logit(np.clip(outer["base"][present], 1e-6, 1 - 1e-6)) + 0.5 * raw_margin)
            candidate = outer["champion"].copy()
            candidate[present] = correction
            if not np.array_equal(candidate, saved["candidate_probability"]):
                raise AssertionError(f"Fixed base-logit + shrinkage formula differs for {key}")
            if not np.array_equal(candidate[outer_missing], outer["champion"][outer_missing]):
                raise AssertionError(f"Missing-stream route changed for {key}")
            verified_vectors[cutoff][head_name] = {"rows": expected_rows,
                                                   "champion": outer["champion"],
                                                   "candidate": candidate,
                                                   "missing": outer_missing}
            head_checks[key] = {"oof_present_rows": int(len(oof["present_rows"])),
                                "outer_present_rows": int(present.sum()),
                                "outer_missing_rows": int(outer_missing.sum()),
                                "raw_margin_sha256": array_digest(raw_margin),
                                "candidate_probability_sha256": array_digest(candidate[present]),
                                "missing_route_exact_match": True}

    evaluation = independent_views(frame, y, cycle, verified_vectors, report["evaluation"])
    all_deltas = [evaluation[view][head]["present_delta_macro_f1"]
                  for view in ("W1", "W4", "H12_E", "H12_L") for head in run.HEADS]
    pooled_improvement = any(evaluation[view][head]["present_delta_macro_f1"] > 0
                             for view in ("W1_plus_W4_pilot_slice", "H12_pooled") for head in run.HEADS)
    gate_pass = all(delta >= -0.01 for delta in all_deltas) and pooled_improvement
    if report["gate"]["overall_pass"] is not gate_pass:
        raise AssertionError("Continuation gate result differs from independent reconstruction")
    result = {"status": "PASS", "verification_version": 2,
              "original_freeze_signature": freeze["signature"],
              "deterministic_replay_head_fits": 4,
              "new_candidate_configurations_or_outer_scores": 0,
              "test_read": False, "kaggle_read_or_submit": False,
              "outer_labels_used_for_fit_or_selection": False,
              "verification": "Rebuilt exact past OOF blocks and head preprocessing; refit same four frozen heads; confirmed fingerprints and model text hashes; reconstructed raw margins and expit(logit(C2_B)+0.5*margin) bit-for-bit; recomputed controls, routed metrics and gate.",
              "head_checks": head_checks,
              "evaluation_views_recomputed": evaluation,
              "gate_pass": gate_pass}
    out_path = run.REPORT_DIR / "verification_v2.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    amendment = {"status": "VERIFIER_ONLY_AMENDMENT_AFTER_PILOT_FITS",
                 "parent_freeze_signature": freeze["signature"],
                 "original_verifier_path": "scripts/optimization_v3_verify_cls_residual.py",
                 "original_verifier_sha256": freeze["identity"]["code_sha256"]["scripts/optimization_v3_verify_cls_residual.py"],
                 "original_verifier_runtime_exit_code": -1073740791,
                 "failure_cause": "LightGBM C++ abort while parsing the archived model_to_string artifact with Booster(model_file); no verification result was written.",
                 "replacement_verifier_path": "scripts/optimization_v3_verify_cls_residual_v2.py",
                 "replacement_verifier_sha256": run.file_sha256(Path(__file__).resolve()),
                 "prediction_artifacts_sha256": {meta["prediction_path"]: meta["prediction_sha256"]
                                                 for meta in report["head_metadata"].values()},
                 "results_sha256": run.file_sha256(result_path),
                 "verification_sha256": run.file_sha256(out_path),
                 "candidate_fit_outputs_changed": False,
                 "refit_count_is_deterministic_verification_only": 4,
                 "new_candidate_configurations_or_outer_scores": 0,
                 "test_read": False,
                 "preserved": ["freeze.json", "freeze_amendment_01.json", "original verifier", "pilot result", "prediction caches"]}
    amendment_path = run.REPORT_DIR / "verification_amendment_01.json"
    amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
