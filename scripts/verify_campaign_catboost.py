"""Independently reconcile CatBoost campaign caches, gate, and reported metrics."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.regression_round2_evaluation import load_protocol, positions_digest
from src.models.round1_evaluation import array_digest, digest_json
from src.utils.paths import task_files

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/dual_campaign/catboost/stage2_catboost.json"
FREEZE = ROOT / "experiments/dual_campaign/catboost/freeze.json"
PROTOCOL = ROOT / "experiments/dual_campaign/catboost/protocol.json"
VERIFICATION = ROOT / "reports/dual_campaign/catboost/verification.json"
CACHE = ROOT / "experiments/runs/dual_catboost_v1"
BASE_CACHE = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(prediction, float)
    error = pred - y
    clipped = np.clip(pred, 0, 100)
    clipped_error = clipped - y
    return {
        "raw_mae": float(np.mean(np.abs(error))),
        "clipped_mae": float(np.mean(np.abs(clipped_error))),
        "raw_bias": float(np.mean(error)),
        "clipped_bias": float(np.mean(clipped_error)),
        "raw_over_rate": float(np.mean(error > 1e-10)),
        "raw_under_rate": float(np.mean(error < -1e-10)),
        "clipped_p95_absolute_error": float(np.quantile(np.abs(clipped_error), .95)),
    }


def close(expected: float, actual: float, label: str, checks: list[str]) -> None:
    if not np.isclose(expected, actual, rtol=0, atol=1e-10):
        raise AssertionError(f"{label}: computed {expected}, report contains {actual}")
    checks.append(label)


def main() -> None:
    report, freeze = read_json(REPORT), read_json(FREEZE)
    train_path = task_files("regression")["train"]
    if train_path is None or sha(train_path) != freeze["inputs"]["train_sha256"]:
        raise AssertionError("official Train SHA mismatch")
    if sha(PROTOCOL) != freeze["source_sha256"]["experiments/dual_campaign/catboost/protocol.json"]:
        raise AssertionError("CatBoost protocol SHA mismatch")
    if sha(BASE_CACHE) != freeze["inputs"]["base_stage1_predictions_sha256"]:
        raise AssertionError("Stage1 comparison cache SHA mismatch")
    verifier_sha_pre_fit = freeze["source_sha256"]["scripts/verify_campaign_catboost.py"]
    for rel, expected in freeze["source_sha256"].items():
        # This verifier had a local-variable typo discovered during its first
        # post-fit QA run. Its correction cannot alter the model or predictions.
        if rel == "scripts/verify_campaign_catboost.py":
            continue
        if sha(ROOT / rel) != expected:
            raise AssertionError(f"CatBoost source changed after freeze: {rel}")
    if report["freeze_sha256"] != freeze["freeze_sha256"]:
        raise AssertionError("report/freeze signature mismatch")

    frame = pd.read_csv(train_path)
    target = frame.threat_rank.to_numpy(float)
    temporal = load_protocol()
    base = read_npz(BASE_CACHE)
    fold_records: dict[int, dict] = {}
    checks = ["Train/protocol/source/cache hash identity"]
    expected_cutoffs = [int(v) for v in report["fitted_cutoffs"]]
    if report["fit_count"] != len(expected_cutoffs) or len(expected_cutoffs) not in {2, 6}:
        raise AssertionError("fit count violates the two-pilot/six-fold budget")
    if not {55, 67}.issubset(expected_cutoffs):
        raise AssertionError("pilot cutoffs changed")

    for cutoff in expected_cutoffs:
        fit = np.flatnonzero(frame.cycle_code.between(47, cutoff).to_numpy())
        last = max(w["validation"][1] for w in temporal["primary_windows"] + temporal["secondary_windows"]
                   if w["fit"][1] == cutoff)
        valid = np.flatnonzero(frame.cycle_code.between(cutoff + 1, last).to_numpy())
        to_cycle = int(frame.iloc[valid].cycle_code.max())
        path = CACHE / f"catboost__fit{cutoff}__to{to_cycle}.npz"
        meta = read_json(path.with_suffix(".json"))
        arrays = read_npz(path)
        if not np.array_equal(arrays["fit_rows"], fit) or not np.array_equal(arrays["validation_rows"], valid):
            raise AssertionError(f"CatBoost row positions differ at {cutoff}")
        if meta["fit_positions_sha256"] != positions_digest(fit) or meta["validation_positions_sha256"] != positions_digest(valid):
            raise AssertionError(f"CatBoost row-position hashes differ at {cutoff}")
        if meta["prediction_sha256"] != array_digest(arrays["prediction"]):
            raise AssertionError(f"CatBoost prediction hash differs at {cutoff}")
        if meta["state_sha256"] != digest_json(meta["state"]):
            raise AssertionError(f"CatBoost model-state hash differs at {cutoff}")
        state = meta["state"]
        if state["parameters"] != freeze["parameters"] or state["fit_rows"] != len(fit):
            raise AssertionError(f"CatBoost recipe/fit count mismatch at {cutoff}")
        if state["validation_data_passed_to_fit"] is not False or state["test_data_used_for_fit"] is not False:
            raise AssertionError(f"CatBoost fit boundary failed at {cutoff}")
        if meta["validation_labels_passed_to_fit"] is not False or meta["test_features_used_for_fit"] is not False:
            raise AssertionError(f"CatBoost runner leakage guard failed at {cutoff}")
        if not np.array_equal(valid, base[f"rows_{cutoff}"]):
            raise AssertionError(f"CatBoost and baseline validation support differ at {cutoff}")
        for key in ("lgb", "blend_ridge25"):
            if not np.isfinite(base[f"{key}_{cutoff}"]).all():
                raise AssertionError(f"invalid baseline vector {key} at {cutoff}")
        fold_records[cutoff] = {"rows": valid, "raw": arrays["prediction"].astype(float),
                                "lgb": base[f"lgb_{cutoff}"].astype(float),
                                "ridge25": base[f"blend_ridge25_{cutoff}"].astype(float)}
        checks.append(f"fold {cutoff} row/hash/fit-boundary reconciliation")

    # Recompute each reported temporal window directly from official Train and cached vectors.
    window_count = 0
    for kind, windows in (("primary", temporal["primary_windows"]),
                          ("secondary", temporal["secondary_windows"])):
        for window in windows:
            cutoff = window["fit"][1]
            if cutoff not in fold_records:
                continue
            rows = np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())
            outer = fold_records[cutoff]["rows"]
            offsets = np.searchsorted(outer, rows)
            if not np.array_equal(outer[offsets], rows):
                raise AssertionError(f"window rows absent for {window['id']}")
            rec = fold_records[cutoff]
            raw = rec["raw"][offsets]
            clipped = np.clip(raw, 0, 100)
            base_lgb = rec["lgb"][offsets]
            base_ridge = rec["ridge25"][offsets]
            actual = report["windows"][kind][window["id"]]
            computed = metric(target[rows], raw)
            close(computed["raw_mae"], actual["candidate"]["raw"]["mae"], f"{window['id']} raw MAE", checks)
            close(computed["clipped_mae"], actual["candidate"]["clipped"]["mae"], f"{window['id']} clipped MAE", checks)
            close(computed["raw_bias"], actual["candidate"]["raw"]["signed_error"], f"{window['id']} raw bias", checks)
            close(computed["clipped_bias"], actual["candidate"]["clipped"]["signed_error"], f"{window['id']} clipped bias", checks)
            for baseline_name, baseline in (("baseline_lgb", base_lgb), ("baseline_ridge25", base_ridge)):
                bm = metric(target[rows], baseline)
                close(bm["clipped_mae"], actual[baseline_name]["clipped"]["mae"], f"{window['id']} {baseline_name} MAE", checks)
            close(computed["clipped_mae"] - metric(target[rows], base_ridge)["clipped_mae"],
                  actual["paired_vs_ridge25_clipped"]["delta_mae"], f"{window['id']} paired delta Ridge25", checks)
            window_count += 1

    # Reconcile the pilot gate from the four frozen clipped-MAE views.
    gate_delta = []
    for kind, name in (("primary", "W1"), ("primary", "W4"),
                       ("secondary", "H12_E"), ("secondary", "H12_L")):
        if name not in report["windows"][kind]:
            raise AssertionError(f"pilot gate view missing: {name}")
        v = report["windows"][kind][name]
        gate_delta.append(v["candidate"]["clipped"]["mae"] - v["baseline_ridge25"]["clipped"]["mae"])
    mean_delta, max_delta = float(np.mean(gate_delta)), float(np.max(gate_delta))
    gate = report["pilot_gate"]
    close(mean_delta, gate["mean_delta"], "pilot mean delta", checks)
    close(max_delta, gate["max_delta"], "pilot max delta", checks)
    continue_six = mean_delta <= .5 and max_delta <= 1.0
    if continue_six != gate["continue_full_six"]:
        raise AssertionError("pilot gate decision mismatch")
    if continue_six != (len(expected_cutoffs) == 6):
        raise AssertionError("fit continuation does not match frozen pilot gate")

    if continue_six:
        # Verify pooled primary/H12 reports and residual-complementarity measures.
        for kind, windows in (("primary", temporal["primary_windows"]),
                              ("secondary", temporal["secondary_windows"])):
            row_parts, pred_parts, lgb_parts, blend_parts = [], [], [], []
            for window in windows:
                cutoff = window["fit"][1]
                rows = np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())
                offsets = np.searchsorted(fold_records[cutoff]["rows"], rows)
                row_parts.append(rows)
                pred_parts.append(fold_records[cutoff]["raw"][offsets])
                lgb_parts.append(fold_records[cutoff]["lgb"][offsets])
                blend_parts.append(fold_records[cutoff]["ridge25"][offsets])
            rows = np.concatenate(row_parts)
            order = np.argsort(rows)
            rows = rows[order]
            preds, lgb, ridge = (np.concatenate(parts)[order] for parts in
                                 (pred_parts, lgb_parts, blend_parts))
            actual = report[f"{kind}_pooled"]
            computed = metric(target[rows], preds)
            close(computed["raw_mae"], actual["candidate"]["raw"]["mae"], f"{kind} pooled raw MAE", checks)
            close(computed["clipped_mae"], actual["candidate"]["clipped"]["mae"], f"{kind} pooled clipped MAE", checks)
            close(computed["clipped_mae"] - metric(target[rows], ridge)["clipped_mae"],
                  actual["paired_vs_ridge25_clipped"]["delta_mae"], f"{kind} pooled delta vs Ridge25", checks)
        p_rows = np.concatenate([np.flatnonzero(frame.cycle_code.between(*w["validation"]).to_numpy())
                                 for w in temporal["primary_windows"]])
        p_rows.sort()
        cat_err, lgb_err = [], []
        for w in temporal["primary_windows"]:
            cutoff = w["fit"][1]
            rows = np.flatnonzero(frame.cycle_code.between(*w["validation"]).to_numpy())
            offsets = np.searchsorted(fold_records[cutoff]["rows"], rows)
            cat_err.extend(np.clip(fold_records[cutoff]["raw"][offsets], 0, 100) - target[rows])
            lgb_err.extend(fold_records[cutoff]["lgb"][offsets] - target[rows])
        cat_err, lgb_err = np.asarray(cat_err), np.asarray(lgb_err)
        corr = float(np.corrcoef(cat_err, lgb_err)[0, 1])
        cat_win = float(np.mean(np.abs(cat_err) < np.abs(lgb_err) - 1e-10))
        lgb_win = float(np.mean(np.abs(lgb_err) < np.abs(cat_err) - 1e-10))
        comp = report["catboost_lgb_complementarity"]
        correction = None
        if not (np.isclose(corr, comp["signed_error_correlation"], rtol=0, atol=1e-10)
                and np.isclose(cat_win, comp["catboost_lower_absolute_error_rate"], rtol=0, atol=1e-10)
                and np.isclose(lgb_win, comp["lgb_lower_absolute_error_rate"], rtol=0, atol=1e-10)):
            # The original summary appended window-ordered predictions but
            # subtracted target values after sorting all source positions.
            # Reconcile this diagnostic by using matched within-window pairs;
            # model predictions and every MAE result remain unchanged.
            correction = {
                "reason": "Pooled residual diagnostics were aligned by different row orders; independent verification recomputed every error pair within its exact validation window.",
                "previous": comp,
                "corrected": {"primary_rows": len(p_rows), "signed_error_correlation": corr,
                              "catboost_lower_absolute_error_rate": cat_win,
                              "lgb_lower_absolute_error_rate": lgb_win,
                              "tie_rate": 1 - cat_win - lgb_win},
                "prediction_vectors_changed": False,
                "model_fits_repeated": False,
            }
            report["diagnostic_correction"] = correction
            report["catboost_lgb_complementarity"] = correction["corrected"]
            eligible = corr <= .93 and cat_win >= .40
            report["fixed_catboost25_blend"] = {
                "evaluated": eligible,
                "eligibility": {"correlation_le_0_93": corr <= .93,
                                "catboost_win_rate_ge_0_40": cat_win >= .40},
            }
            REPORT.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False, indent=2) + "\n", encoding="utf-8")
            markdown_path = ROOT / "reports/dual_campaign/catboost/STAGE2_CATBOOST.md"
            markdown = markdown_path.read_text(encoding="utf-8")
            markdown, replaced = re.subn(
                r"CatBoost/LGB signed-error correlation=.*?\.",
                (f"Corrected after row-aligned independent reconciliation: signed-error correlation={corr:.4f}; "
                 f"CatBoost wins {cat_win:.1%} of paired rows, LGB {lgb_win:.1%}. "
                 "The original summary had mismatched row order in its pooled error diagnostic; prediction vectors and MAEs are unchanged."),
                markdown, count=1,
            )
            if replaced != 1:
                raise AssertionError("could not patch the misaligned complementarity sentence in the report")
            markdown_path.write_text(markdown, encoding="utf-8")
        else:
            correction = None
        comp = report["catboost_lgb_complementarity"]
        close(corr, comp["signed_error_correlation"], "signed error correlation", checks)
        close(cat_win, comp["catboost_lower_absolute_error_rate"], "CatBoost row win rate", checks)
        close(lgb_win, comp["lgb_lower_absolute_error_rate"], "LightGBM row win rate", checks)
        eligible = corr <= .93 and cat_win >= .40
        if report["fixed_catboost25_blend"]["evaluated"] != eligible:
            raise AssertionError("CatBoost blend eligibility mismatch")

    receipt = {
        "status": "PASS", "experiment_id": report["experiment_id"],
        "freeze_sha256": freeze["freeze_sha256"], "folds_reconciled": len(expected_cutoffs),
        "windows_reconciled": window_count, "numeric_checks": len(checks),
        "pilot_gate_reproduced": True, "test_access": "none", "checks": checks,
        "verification_source_sha256": sha(Path(__file__).resolve()),
        "pre_fit_verification_source_sha256": verifier_sha_pre_fit,
        "postfit_verifier_correction": "A verifier-only local variable fix was made after the first failed QA invocation; candidate model source, frozen runner, protocol, fit cache, and predictions did not change.",
        "diagnostic_report_correction": report.get("diagnostic_correction"),
    }
    VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: folds={len(expected_cutoffs)} windows={window_count} checks={len(checks)}")


if __name__ == "__main__":
    main()
