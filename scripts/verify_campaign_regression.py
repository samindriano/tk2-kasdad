"""Independent reconciliation of the saved Stage 1 regression blend report."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.regression_round2_evaluation import load_protocol, positions_digest
from src.models.round1_evaluation import array_digest, digest_json
from src.utils.paths import task_files

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/dual_campaign/regression/stage1_paired_blend.json"
FREEZE = ROOT / "experiments/dual_campaign/regression/freeze.json"
PROTOCOL = ROOT / "experiments/dual_campaign/regression/protocol.json"
VERIFICATION = ROOT / "reports/dual_campaign/regression/verification.json"
PREDICTIONS = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
HIST = ROOT / "experiments/runs/regression_research_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def pct(pred: np.ndarray, cycle: np.ndarray, weight: float) -> np.ndarray:
    result = np.asarray(pred, dtype=float).copy()
    if weight:
        rank = np.empty(len(result), dtype=float)
        for c in pd.unique(cycle):
            mask = cycle == c
            rank[mask] = rankdata(result[mask], method="average") * 100.0 / mask.sum()
        result = (1.0 - weight) * result + weight * rank
    return np.clip(result, 0, 100)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def compare_metric(expected: float, actual: float, name: str, checks: list[str]) -> None:
    if not np.isclose(expected, actual, rtol=0, atol=1e-10):
        raise AssertionError(f"{name}: expected {expected}, reported {actual}")
    checks.append(name)


def main() -> None:
    report, freeze = json_read(REPORT), json_read(FREEZE)
    train_path = task_files("regression")["train"]
    assert train_path is not None
    if sha256(train_path) != freeze["official_regression_train_sha256"]:
        raise AssertionError("Train SHA mismatch")
    if sha256(PROTOCOL) != freeze["protocol_sha256"]:
        raise AssertionError("protocol SHA mismatch")
    if sha256(PREDICTIONS) != report["derived_prediction_cache"]["sha256"]:
        raise AssertionError("derived prediction cache SHA mismatch")
    if sha256(PREDICTIONS) != report["derived_prediction_cache"]["sha256"] or not report["derived_prediction_cache"]["git_ignored"]:
        raise AssertionError("private prediction cache guard failed")

    f = pd.read_csv(train_path)
    y, cycle = f.threat_rank.to_numpy(float), f.cycle_code.to_numpy()
    protocol = load_protocol()
    saved = read_npz(PREDICTIONS)
    lgb_stage = json_read(ROOT / "reports/regression_research/stage5_lgb_text.json")
    ridge_stage = json_read(ROOT / "reports/regression_research/stage23_models.json")
    checks: list[str] = ["Train/protocol/prediction cache hashes"]
    folds = {}
    for cutoff in (55, 59, 63, 67, 71, 75):
        fit_inner = np.flatnonzero(f.cycle_code.between(47, cutoff - 4).to_numpy())
        valid_inner = np.flatnonzero(f.cycle_code.between(cutoff - 3, cutoff).to_numpy())
        last = max(w["validation"][1] for w in protocol["primary_windows"] + protocol["secondary_windows"]
                   if w["fit"][1] == cutoff)
        outer = np.flatnonzero(f.cycle_code.between(cutoff + 1, last).to_numpy())
        assert np.array_equal(saved[f"rows_{cutoff}"], outer)
        lgb_choice = next(v["selected"] for v in lgb_stage["inner"] if v["fit_end"] == cutoff)
        ridge_choice = next(v["selected"] for v in ridge_stage["inner"]
                            if v["fit_end"] == cutoff and v["recipe"] == "hybrid_ridge")
        lgb_inner_path = HIST / f"lgbtext__{lgb_choice['param']}__fit{cutoff-4}__to{cutoff}.npz"
        ridge_inner_path = HIST / f"model__hybrid_ridge__base__fit{cutoff-4}__to{cutoff}.npz"
        for path, fit, valid, label in ((lgb_inner_path, fit_inner, valid_inner, "LGB"),
                                         (ridge_inner_path, fit_inner, valid_inner, "Ridge")):
            arrays = read_npz(path)
            meta = json_read(path.with_suffix(".json"))
            if not np.array_equal(arrays["fit_rows"], fit) or not np.array_equal(arrays["validation_rows"], valid):
                raise AssertionError(f"{label} inner rows mismatch at {cutoff}")
            if meta["fit_positions_sha256"] != positions_digest(fit) or meta["validation_positions_sha256"] != positions_digest(valid):
                raise AssertionError(f"{label} inner position hash mismatch at {cutoff}")
            pred = arrays["prediction"].astype(float)
            if meta["prediction_sha256"] != array_digest(pred) or meta["state_sha256"] != digest_json(meta["state"]):
                raise AssertionError(f"{label} inner state/prediction hash mismatch at {cutoff}")
            if meta["validation_labels_passed_to_fit"] is not False:
                raise AssertionError(f"{label} inner labels touched fit at {cutoff}")
        lgb_outer_raw = read_npz(HIST / f"eval_lgbtext__fit{cutoff}__raw.npz")
        ridge_outer_raw = read_npz(HIST / f"eval__fit{cutoff}__hybrid_ridge__raw.npz")
        lgb_outer_mapped = read_npz(HIST / f"eval_lgbtext__fit{cutoff}__inner.npz")
        ridge_outer_mapped = read_npz(HIST / f"eval__fit{cutoff}__hybrid_ridge__inner.npz")
        for arrays in (lgb_outer_raw, ridge_outer_raw, lgb_outer_mapped, ridge_outer_mapped):
            if not np.array_equal(arrays["validation_rows"], outer):
                raise AssertionError(f"outer rows mismatch at {cutoff}")
        if not np.array_equal(lgb_outer_mapped["prediction"], saved[f"lgb_{cutoff}"]):
            raise AssertionError(f"LGB outer mismatch with derived cache at {cutoff}")
        if not np.array_equal(ridge_outer_mapped["prediction"], saved[f"ridge_{cutoff}"]):
            raise AssertionError(f"Ridge outer mismatch with derived cache at {cutoff}")
        for component, raw, mapped, weight in (("LGB", lgb_outer_raw["prediction"], lgb_outer_mapped["prediction"], lgb_choice["rank_weight"]),
                                                ("Ridge", ridge_outer_raw["prediction"], ridge_outer_mapped["prediction"], ridge_choice["rank_weight"])):
            if not np.array_equal(pct(raw, cycle[outer], weight), mapped):
                raise AssertionError(f"{component} outer policy mismatch at {cutoff}")

        inner_lgb = pct(read_npz(lgb_inner_path)["prediction"], cycle[valid_inner], lgb_choice["rank_weight"])
        inner_ridge = pct(read_npz(ridge_inner_path)["prediction"], cycle[valid_inner], ridge_choice["rank_weight"])
        grid = []
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            blended = (1 - alpha) * inner_lgb + alpha * inner_ridge
            grid.append((alpha, mae(y[valid_inner], blended)))
        minimum = min(score for _, score in grid)
        chosen = min((w for w, score in grid if score <= minimum + 1e-10))
        saved_selection = report["weight_selection"][str(cutoff)]["ridge_share"]
        if chosen != saved_selection["selected"]:
            raise AssertionError(f"inner selected alpha mismatch at {cutoff}")
        for (w, score), record in zip(grid, saved_selection["grid"]):
            if float(record["weight"]) != w or not np.isclose(record["mae"], score, atol=1e-10, rtol=0):
                raise AssertionError(f"inner alpha grid mismatch at {cutoff}/{w}")
        folds[cutoff] = {"outer": outer, "selected": chosen}
        checks.extend([f"{cutoff} fold row/hash/mapping/inner-selection reconciliation"])

    candidate_names = ("lgb_text_inner_v1", "ridge_25", "ridge_50", "hybrid_ridge_inner_v1", "inner_selected")
    metrics_checked = 0
    for name in candidate_names:
        for kind, windows in (("primary", protocol["primary_windows"]),
                              ("secondary", protocol["secondary_windows"])):
            for window in windows:
                cutoff = window["fit"][1]
                rows = np.flatnonzero(f.cycle_code.between(*window["validation"]).to_numpy())
                offset = np.searchsorted(folds[cutoff]["outer"], rows)
                lgb = saved[f"lgb_{cutoff}"][offset]
                ridge = saved[f"ridge_{cutoff}"][offset]
                alpha = {"lgb_text_inner_v1": 0.0, "ridge_25": .25, "ridge_50": .5,
                         "hybrid_ridge_inner_v1": 1.0,
                         "inner_selected": folds[cutoff]["selected"]}[name]
                prediction = np.clip((1 - alpha) * lgb + alpha * ridge, 0, 100)
                actual = report["results"][name][kind][window["id"]]["candidate"]
                compare_metric(mae(y[rows], prediction), actual["raw"]["mae"], f"{name}/{window['id']} MAE", checks)
                compare_metric(float(np.mean(np.abs(prediction - y[rows]))), actual["clipped"]["mae"], f"{name}/{window['id']} clipped MAE", checks)
                compare_metric(float(np.mean(prediction - y[rows])), actual["raw"]["signed_error"], f"{name}/{window['id']} bias", checks)
                cycle_errors = [mae(y[rows][cycle[rows] == c], prediction[cycle[rows] == c])
                                for c in pd.unique(cycle[rows])]
                compare_metric(float(np.mean(cycle_errors)), actual["equal_cycle_mae"], f"{name}/{window['id']} equal cycle", checks)
                metrics_checked += 4
        for kind, windows in (("primary", protocol["primary_windows"]),
                              ("secondary", protocol["secondary_windows"])):
            row_parts, pred_parts, control_parts = [], [], []
            for window in windows:
                cutoff = window["fit"][1]
                rows_part = np.flatnonzero(f.cycle_code.between(*window["validation"]).to_numpy())
                offsets = np.searchsorted(folds[cutoff]["outer"], rows_part)
                lgb_part = saved[f"lgb_{cutoff}"][offsets]
                ridge_part = saved[f"ridge_{cutoff}"][offsets]
                alpha = {"lgb_text_inner_v1": 0.0, "ridge_25": .25, "ridge_50": .5,
                         "hybrid_ridge_inner_v1": 1.0,
                         "inner_selected": folds[cutoff]["selected"]}[name]
                row_parts.append(rows_part)
                pred_parts.append(np.clip((1 - alpha) * lgb_part + alpha * ridge_part, 0, 100))
                control_parts.append(lgb_part)
            rows = np.concatenate(row_parts)
            preds, controls = np.concatenate(pred_parts), np.concatenate(control_parts)
            order = np.argsort(rows)
            rows, preds, controls = rows[order], preds[order], controls[order]
            actual = report["results"][name][kind + "_pooled"]["candidate"]
            compare_metric(mae(y[rows], preds), actual["raw"]["mae"], f"{name}/{kind} pooled MAE", checks)
            compare_metric(float(np.mean(preds - y[rows])), actual["raw"]["signed_error"], f"{name}/{kind} pooled bias", checks)
            compare_metric(mae(y[rows], preds) - mae(y[rows], controls),
                           report["results"][name][kind + "_pooled"]["paired_vs_lgb"]["delta_mae"],
                           f"{name}/{kind} paired delta", checks)
            metrics_checked += 3

    receipt = {
        "status": "PASS", "experiment_id": report["experiment_id"],
        "freeze_sha256": freeze["freeze_sha256"],
        "source_rows": {"primary": report["source_rows_used"]["primary"],
                        "h12": report["source_rows_used"]["h12"]},
        "folds_reconciled": 6, "candidate_recipes_reconciled": len(candidate_names),
        "numeric_metrics_reconciled": metrics_checked,
        "independent_checks": checks,
        "test_access": "none",
    }
    VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
    VERIFICATION.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: folds=6 candidates=5 numeric_metrics={metrics_checked} checks={len(checks)}")


if __name__ == "__main__":
    main()
