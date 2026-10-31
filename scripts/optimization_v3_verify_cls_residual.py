"""Independent rebuild and raw-margin verification for the two-head pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
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


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Saved scores are not finite probabilities aligned to official Train")
    return {"n": int(len(y)), "macro_f1": float(f1_score(y, p >= 0.5, average="macro", zero_division=0)),
            "error_count": int(np.count_nonzero((p >= 0.5) != y))}


def close_metrics(observed: dict, expected: dict) -> None:
    for key, value in expected.items():
        if isinstance(value, float):
            if not np.isclose(observed[key], value, rtol=0, atol=1e-12):
                raise AssertionError(f"Metric {key} differs: {observed[key]} != {value}")
        elif observed[key] != value:
            raise AssertionError(f"Metric {key} differs: {observed[key]} != {value}")


def verify() -> dict:
    freeze = run.check_freeze()
    result_path = run.REPORT_DIR / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError("Pilot results are absent; run the frozen pilot first")
    report = run.read_json(result_path)
    if report.get("freeze_signature") != freeze["signature"] or report.get("outer_labels_used_for_fit_or_selection") is not False:
        raise AssertionError("Pilot result freeze identity or label-boundary attestation differs")
    frame, y, cycle, missing, _ = run.load_train()
    checked_heads = 0
    verified_scores = {}
    for cutoff in run.OUTER:
        outer = run.outer_inputs(frame, y, cycle, missing, cutoff)
        for head_name, score_names in run.HEADS.items():
            key = f"{cutoff}/{head_name}"
            meta = report["head_metadata"][key]
            oof = run.prepare_oof(frame, y, cycle, missing, cutoff, head_name)
            if meta["oof_raw_input_sha256"] != oof["raw_input_sha256"]:
                raise AssertionError(f"Raw OOF input hash differs for {key}")
            if meta["oof_rows_present"] != len(oof["present_rows"]) or meta["state"]["rows"] != len(oof["present_rows"]):
                raise AssertionError(f"OOF fit support differs for {key}")
            if meta["state"]["score_names"] != list(score_names) or meta["state"]["input_features"] != list(run.EXPANDED_FEATURES):
                raise AssertionError(f"Head allowlist or score identity differs for {key}")
            path = run.ROOT / meta["prediction_path"]
            model_path = run.ROOT / meta["model_path"]
            if run.file_sha256(path) != meta["prediction_sha256"] or run.file_sha256(model_path) != meta["model_sha256"]:
                raise AssertionError(f"Prediction/model artifact hash mismatch for {key}")
            with np.load(path, allow_pickle=False) as z:
                saved = {name: z[name].copy() for name in z.files}
            expected_rows = outer["rows"]
            if not np.array_equal(saved["outer_rows"], expected_rows):
                raise AssertionError(f"Outer row IDs differ for {key}")
            outer_missing = missing[expected_rows]
            if not np.array_equal(saved["missing_mask"].astype(bool), outer_missing):
                raise AssertionError(f"Stream route mask differs for {key}")
            if not np.array_equal(saved["base_probability"], outer["base"]) or not np.array_equal(saved["cb_probability"], outer["cb"]):
                raise AssertionError(f"Cached control score vectors differ for {key}")
            if not np.array_equal(saved["champion_probability"], outer["champion"]):
                raise AssertionError(f"Current champion route changed for {key}")
            if not np.array_equal(saved["oof_present_rows"], oof["present_rows"]):
                raise AssertionError(f"Meta-training row identity differs for {key}")
            if not np.array_equal(saved["oof_scores"], oof["scores"]) or not np.array_equal(saved["oof_target"], oof["y"]):
                raise AssertionError(f"Meta-training score or label vectors differ for {key}")

            # Refit exactly the frozen model from the raw past OOF inputs; this
            # independently checks the preprocessing state and booster identity.
            rebuilt = OOFResidualHead("classification", score_names)
            with threadpool_limits(limits=2):
                rebuilt.fit(oof["X"], oof["scores"], oof["y"])
            rebuilt_state = rebuilt.audit_state()
            for field in ("fit_data_sha256", "fit_scores_sha256", "target_sha256", "booster_sha256"):
                if rebuilt_state[field] != meta["state"][field]:
                    raise AssertionError(f"Refitted {key} state hash differs at {field}")
            if rebuilt_state["rows"] != meta["state"]["rows"] or rebuilt_state["preprocessor"] != meta["state"]["preprocessor"]:
                raise AssertionError(f"Fit-only preprocessor state differs for {key}")
            outer_present = ~outer_missing
            X_outer = frame.iloc[expected_rows[outer_present]][list(run.EXPANDED_FEATURES)].copy()
            score_indices = [0] if head_name == "A_base_structured" else [0, 1]
            outer_scores = np.column_stack([outer["base"], outer["cb"]])[outer_present][:, score_indices]
            design, _, _ = rebuilt._design(X_outer, outer_scores, False)
            design_array = design.to_numpy(dtype=float)
            if not np.array_equal(design_array, saved["outer_design_present"]):
                raise AssertionError(f"Rebuilt outer design differs for {key}")
            if array_digest(design_array) != meta["outer_design_sha256"]:
                raise AssertionError(f"Outer design hash differs for {key}")
            booster = lgb.Booster(model_file=str(model_path))
            raw_margin = booster.predict(design, raw_score=True, num_threads=2)
            if not np.allclose(raw_margin, saved["raw_correction_present"], rtol=0, atol=1e-12):
                raise AssertionError(f"Serialized booster raw margin differs for {key}")
            if array_digest(raw_margin) != meta["raw_correction_sha256"]:
                raise AssertionError(f"Raw correction digest differs for {key}")
            corrected = expit(logit(np.clip(outer["base"][outer_present], 1e-6, 1 - 1e-6)) + 0.5 * raw_margin)
            expected_candidate = outer["champion"].copy()
            expected_candidate[outer_present] = corrected
            if not np.allclose(expected_candidate, saved["candidate_probability"], rtol=0, atol=1e-12):
                raise AssertionError(f"Base-logit + fixed-shrinkage reconstruction differs for {key}")
            if not np.array_equal(saved["candidate_probability"][outer_missing], saved["champion_probability"][outer_missing]):
                raise AssertionError(f"Missing-stream route changed for {key}")
            for field in ("base_probability", "cb_probability", "champion_probability", "candidate_probability"):
                if not np.isfinite(saved[field]).all():
                    raise AssertionError(f"Nonfinite vector {field} for {key}")
            verified_scores[key] = {"outer_rows": len(expected_rows),
                                    "present_rows": int(outer_present.sum()),
                                    "missing_rows": int(outer_missing.sum()),
                                    "current_present": metrics(y[expected_rows][outer_present], outer["champion"][outer_present]),
                                    "candidate_present": metrics(y[expected_rows][outer_present], expected_candidate[outer_present]),
                                    "missing_exact_match": True,
                                    "design_sha256": array_digest(design_array),
                                    "raw_margin_sha256": array_digest(raw_margin)}
            checked_heads += 1

    # Independently reconstruct the reported evaluation slices from verified
    # vectors and official Train labels.
    rebuilt_records = {}
    for cutoff in run.OUTER:
        rebuilt_records[cutoff] = {}
        for head_name in run.HEADS:
            meta = report["head_metadata"][f"{cutoff}/{head_name}"]
            with np.load(run.ROOT / meta["prediction_path"], allow_pickle=False) as z:
                arr = {name: z[name].copy() for name in z.files}
            rebuilt_records[cutoff][head_name] = {
                "outer_rows": arr["outer_rows"],
                "champion_probability": arr["champion_probability"],
                "candidate_probability": arr["candidate_probability"],
                "missing_mask": arr["missing_mask"].astype(bool),
            }
    rebuilt_eval = run.compare_views(frame, y, cycle, rebuilt_records)
    for view, heads in rebuilt_eval.items():
        for head_name, metrics_view in heads.items():
            expected = report["evaluation"][view][head_name]
            close_metrics(metrics_view["present_current"], expected["present_current"])
            close_metrics(metrics_view["present_candidate"], expected["present_candidate"])
            close_metrics(metrics_view["all_current"], expected["all_current"])
            close_metrics(metrics_view["all_candidate"], expected["all_candidate"])
            if metrics_view["missing_exact_match"] is not True or expected["missing_exact_match"] is not True:
                raise AssertionError(f"Missing route comparison fails in {view}/{head_name}")
    result = {"status": "PASS", "freeze_signature": freeze["signature"],
              "checked_head_fits": checked_heads,
              "outer_labels_used_for_fit_or_selection": False,
              "test_read": False, "kaggle_read_or_submit": False,
              "reconstruction": "OOF rows/scores/features refit independently; serialized booster raw margin checked; corrected probability reconstructed from base logit plus 0.5 raw margin; official Train metrics recomputed.",
              "heads": verified_scores,
              "evaluation_views_recomputed": list(rebuilt_eval)}
    out = run.REPORT_DIR / "verification.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
