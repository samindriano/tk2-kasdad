"""Run a frozen, bounded CatBoost MAE temporal challenger (never opens Test)."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.campaign_catboost import CATBOOST_PARAMETERS, build_catboost_campaign
from src.models.regression_round2_evaluation import load_protocol, positions_digest
from src.models.regression_round2_metrics import paired, summarize
from src.models.round1_evaluation import array_digest, digest_json
from src.utils.paths import task_files

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/dual_campaign/catboost"
PROTOCOL_PATH = EXP / "protocol.json"
FREEZE_PATH = EXP / "freeze.json"
REPORT_DIR = ROOT / "reports/dual_campaign/catboost"
CACHE = ROOT / "experiments/runs/dual_catboost_v1"
STAGE1_CACHE = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
LGB_REPORT = ROOT / "reports/regression_research/stage5_lgb_text.json"
ROOT_PROTOCOL = ROOT / "experiments/regression_round2/protocol.json"
FOLD_ENDS = (55, 59, 63, 67, 71, 75)
PILOT_ENDS = (55, 67)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, indent=2) + "\n", encoding="utf-8")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def write_freeze(protocol: dict, train_path: Path, base_cache: Path) -> dict:
    import catboost
    temporal = load_protocol()
    stage5 = read_json(ROOT / "experiments/regression_research/stage5_lgb_text_freeze.json")
    stage23 = read_json(ROOT / "experiments/regression_research/stage23_models_freeze.json")
    sources = {
        "src/models/campaign_catboost.py": sha(ROOT / "src/models/campaign_catboost.py"),
        "scripts/research_campaign_catboost.py": sha(Path(__file__).resolve()),
        "scripts/verify_campaign_catboost.py": sha(ROOT / "scripts/verify_campaign_catboost.py"),
        "tests/test_campaign_catboost.py": sha(ROOT / "tests/test_campaign_catboost.py"),
        "experiments/dual_campaign/catboost/protocol.json": sha(PROTOCOL_PATH),
    }
    for rel, expected in stage5["identity"]["code_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"historical LGB model identity changed: {rel}")
    for rel, expected in stage23["identity"]["code_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"historical Ridge model identity changed: {rel}")
    inputs = {
        "train_sha256": sha(train_path),
        "base_stage1_predictions_sha256": sha(base_cache),
        "stage5_report_sha256": sha(LGB_REPORT),
        "stage5_freeze_signature": stage5["signature"],
        "stage23_freeze_signature": stage23["signature"],
        "temporal_protocol_sha256": sha(ROOT_PROTOCOL),
    }
    payload = {
        "experiment_id": protocol["experiment_id"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "catboost_version": catboost.__version__,
        "parameters": dict(CATBOOST_PARAMETERS),
        "inputs": inputs,
        "source_sha256": sources,
        "folds": list(FOLD_ENDS),
        "pilot_folds": list(PILOT_ENDS),
        "max_model_fits": 6,
        "no_test_access": True,
    }
    payload["freeze_sha256"] = digest_json(payload)
    if FREEZE_PATH.exists():
        old = read_json(FREEZE_PATH)
        if old != payload:
            raise RuntimeError("CatBoost freeze changed; create a new experiment id before training")
        return old
    write_json(FREEZE_PATH, payload)
    return payload


def fit_predict(cutoff: int, frame: pd.DataFrame, fit_rows: np.ndarray,
                valid_rows: np.ndarray, freeze: dict) -> tuple[np.ndarray, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    base = CACHE / f"catboost__fit{cutoff}__to{int(frame.iloc[valid_rows].cycle_code.max())}"
    npz_path, json_path = base.with_suffix(".npz"), base.with_suffix(".json")
    key = digest_json({"freeze": freeze["freeze_sha256"], "cutoff": cutoff,
                       "fit": positions_digest(fit_rows), "valid": positions_digest(valid_rows)})
    if npz_path.exists() != json_path.exists():
        raise RuntimeError("CatBoost cache pair is incomplete")
    if npz_path.exists():
        arrays, meta = read_npz(npz_path), read_json(json_path)
        if meta.get("key") != key:
            raise RuntimeError(f"CatBoost cache key mismatch at cutoff {cutoff}")
        if not np.array_equal(arrays["fit_rows"], fit_rows) or not np.array_equal(arrays["validation_rows"], valid_rows):
            raise RuntimeError(f"CatBoost cached positions mismatch at cutoff {cutoff}")
        pred = arrays["prediction"].astype(float)
        if array_digest(pred) != meta.get("prediction_sha256") or meta.get("state_sha256") != digest_json(meta.get("state")):
            raise RuntimeError(f"CatBoost cached model state or prediction hash mismatch at cutoff {cutoff}")
        return pred, meta

    model = build_catboost_campaign()
    features = list(model.feature_names)
    start = perf_counter()
    y = frame.iloc[fit_rows].threat_rank.to_numpy(copy=True)
    with threadpool_limits(limits=2):
        model.fit(frame.iloc[fit_rows][features].copy(), y)
        state = model.audit_state()
        pred = model.predict(frame.iloc[valid_rows][features].copy())
    if model.audit_state() != state:
        raise RuntimeError("CatBoost fit state changed during prediction")
    if len(pred) != len(valid_rows) or not np.isfinite(pred).all():
        raise RuntimeError("CatBoost output is invalid")
    meta = {
        "key": key, "cutoff": cutoff, "parameters": dict(CATBOOST_PARAMETERS),
        "fit_rows": len(fit_rows), "validation_rows": len(valid_rows),
        "fit_positions_sha256": positions_digest(fit_rows),
        "validation_positions_sha256": positions_digest(valid_rows),
        "state": state, "state_sha256": digest_json(state),
        "prediction_sha256": array_digest(pred), "seconds": perf_counter() - start,
        "validation_labels_passed_to_fit": False, "test_features_used_for_fit": False,
    }
    np.savez_compressed(npz_path, fit_rows=fit_rows, validation_rows=valid_rows, prediction=pred)
    write_json(json_path, meta)
    print(f"fit CatBoost cutoff={cutoff} rows={len(fit_rows)} secs={meta['seconds']:.2f}", flush=True)
    return pred, meta


def metric(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return summarize(frame.reset_index(drop=True), prediction, include_impact=False)


def paired_metric(frame: pd.DataFrame, candidate: np.ndarray, control: np.ndarray) -> dict:
    return paired(frame.reset_index(drop=True), candidate, control)


def evaluate_windows(frame: pd.DataFrame, temporal: dict, predictions: dict[int, dict],
                     base_predictions: dict[int, dict]) -> dict:
    result = {"primary": {}, "secondary": {}}
    for kind, windows in (("primary", temporal["primary_windows"]),
                          ("secondary", temporal["secondary_windows"])):
        for window in windows:
            cutoff = window["fit"][1]
            if cutoff not in predictions:
                continue
            valid = np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())
            outer = predictions[cutoff]["rows"]
            offsets = np.searchsorted(outer, valid)
            if not np.array_equal(outer[offsets], valid):
                raise RuntimeError(f"CatBoost cache does not include {window['id']}")
            raw = predictions[cutoff]["prediction"][offsets]
            clipped = np.clip(raw, 0, 100)
            lgb = base_predictions[cutoff]["lgb"][offsets]
            ridge25 = base_predictions[cutoff]["ridge25"][offsets]
            result[kind][window["id"]] = {
                "window_id": window["id"], "cutoff": cutoff, "n": len(valid),
                "candidate": metric(frame.iloc[valid], raw),
                "paired_vs_lgb_clipped": paired_metric(frame.iloc[valid], clipped, lgb),
                "paired_vs_ridge25_clipped": paired_metric(frame.iloc[valid], clipped, ridge25),
                "baseline_lgb": metric(frame.iloc[valid], lgb),
                "baseline_ridge25": metric(frame.iloc[valid], ridge25),
                "clipped_candidate_prediction_sha256": array_digest(clipped),
            }
    return result


def aggregate_windows(frame: pd.DataFrame, temporal: dict, predictions: dict[int, dict],
                      base_predictions: dict[int, dict], kind: str) -> dict:
    windows = temporal["primary_windows"] if kind == "primary" else temporal["secondary_windows"]
    rows_parts, raw_parts, lgb_parts, blend_parts = [], [], [], []
    for window in windows:
        cutoff = window["fit"][1]
        valid = np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())
        outer = predictions[cutoff]["rows"]
        offsets = np.searchsorted(outer, valid)
        rows_parts.append(valid)
        raw_parts.append(predictions[cutoff]["prediction"][offsets])
        lgb_parts.append(base_predictions[cutoff]["lgb"][offsets])
        blend_parts.append(base_predictions[cutoff]["ridge25"][offsets])
    rows = np.concatenate(rows_parts)
    order = np.argsort(rows)
    rows = rows[order]
    raw = np.concatenate(raw_parts)[order]
    lgb = np.concatenate(lgb_parts)[order]
    blend = np.concatenate(blend_parts)[order]
    if len(rows) != len(np.unique(rows)):
        raise RuntimeError(f"duplicate validation rows in {kind}")
    clipped = np.clip(raw, 0, 100)
    return {
        "n": len(rows), "candidate": metric(frame.iloc[rows], raw),
        "paired_vs_lgb_clipped": paired_metric(frame.iloc[rows], clipped, lgb),
        "paired_vs_ridge25_clipped": paired_metric(frame.iloc[rows], clipped, blend),
        "baseline_lgb": metric(frame.iloc[rows], lgb),
        "baseline_ridge25": metric(frame.iloc[rows], blend),
        "prediction_sha256": array_digest(raw),
    }


def main() -> None:
    protocol = read_json(PROTOCOL_PATH)
    temporal = load_protocol()
    train_path = task_files("regression")["train"]
    if train_path is None:
        raise FileNotFoundError("official Regression Train.csv is missing")
    if not STAGE1_CACHE.is_file():
        raise FileNotFoundError("verified Stage1 blend prediction cache is missing")
    freeze = write_freeze(protocol, train_path, STAGE1_CACHE)
    frame = pd.read_csv(train_path)
    if len(frame) != 50608 or frame.cycle_code.min() != 47 or frame.cycle_code.max() != 79:
        raise RuntimeError("official Regression Train shape/cycle support changed")
    base_arrays = read_npz(STAGE1_CACHE)
    predictions: dict[int, dict] = {}
    metas: dict[str, dict] = {}

    def run_cutoff(cutoff: int) -> None:
        fit_rows = np.flatnonzero(frame.cycle_code.between(47, cutoff).to_numpy())
        last = max(w["validation"][1] for w in temporal["primary_windows"] + temporal["secondary_windows"]
                   if w["fit"][1] == cutoff)
        valid_rows = np.flatnonzero(frame.cycle_code.between(cutoff + 1, last).to_numpy())
        if not (frame.iloc[fit_rows].cycle_code.max() == cutoff and
                frame.iloc[valid_rows].cycle_code.min() == cutoff + 1):
            raise RuntimeError(f"CatBoost temporal boundary failed at cutoff {cutoff}")
        raw, meta = fit_predict(cutoff, frame, fit_rows, valid_rows, freeze)
        predictions[cutoff] = {"rows": valid_rows, "prediction": raw}
        metas[str(cutoff)] = meta

    for cutoff in PILOT_ENDS:
        run_cutoff(cutoff)
    base_predictions: dict[int, dict] = {}
    for cutoff in PILOT_ENDS:
        base_predictions[cutoff] = {
            "lgb": base_arrays[f"lgb_{cutoff}"],
            "ridge25": base_arrays[f"blend_ridge25_{cutoff}"],
            "rows": base_arrays[f"rows_{cutoff}"],
        }
        if not np.array_equal(predictions[cutoff]["rows"], base_predictions[cutoff]["rows"]):
            raise RuntimeError(f"Stage1 baseline cache positions differ at {cutoff}")
    pilot_eval = evaluate_windows(frame, temporal, predictions, base_predictions)
    pilot_views = [("primary", "W1"), ("primary", "W4"),
                   ("secondary", "H12_E"), ("secondary", "H12_L")]
    gate_deltas = []
    for kind, name in pilot_views:
        row = pilot_eval[kind][name]
        cand = row["candidate"]["clipped"]["mae"]
        base = row["baseline_ridge25"]["clipped"]["mae"]
        gate_deltas.append({"view": name, "catboost_clipped_mae": cand,
                            "ridge25_mae": base, "delta": cand - base})
    mean_delta = float(np.mean([v["delta"] for v in gate_deltas]))
    max_delta = float(max(v["delta"] for v in gate_deltas))
    continue_full = mean_delta <= 0.50 and max_delta <= 1.00
    if continue_full:
        for cutoff in FOLD_ENDS:
            if cutoff not in predictions:
                run_cutoff(cutoff)
                base_predictions[cutoff] = {
                    "lgb": base_arrays[f"lgb_{cutoff}"],
                    "ridge25": base_arrays[f"blend_ridge25_{cutoff}"],
                    "rows": base_arrays[f"rows_{cutoff}"],
                }
                if not np.array_equal(predictions[cutoff]["rows"], base_predictions[cutoff]["rows"]):
                    raise RuntimeError(f"Stage1 baseline cache positions differ at {cutoff}")
    window_results = evaluate_windows(frame, temporal, predictions, base_predictions)
    report: dict = {
        "experiment_id": protocol["experiment_id"], "freeze_sha256": freeze["freeze_sha256"],
        "source_commit": freeze["source_commit"], "catboost_version": freeze["catboost_version"],
        "parameters": CATBOOST_PARAMETERS, "fitted_cutoffs": sorted(predictions),
        "fit_count": len(predictions), "pilot_gate": {
            "metric": "clipped MAE; fixed and frozen before fitting",
            "criterion": "mean delta vs Ridge25 <= +0.50, maximum delta <= +1.00 across W1/W4/H12_E/H12_L",
            "views": gate_deltas, "mean_delta": mean_delta, "max_delta": max_delta,
            "continue_full_six": continue_full,
        },
        "outer_model_audits": metas,
        "windows": window_results,
        "primary_pooled": None,
        "secondary_pooled": None,
        "catboost_lgb_complementarity": None,
        "fixed_catboost25_blend": None,
        "no_test_access": True,
    }
    if continue_full:
        report["primary_pooled"] = aggregate_windows(frame, temporal, predictions, base_predictions, "primary")
        report["secondary_pooled"] = aggregate_windows(frame, temporal, predictions, base_predictions, "secondary")
        primary_rows = np.concatenate([np.flatnonzero(frame.cycle_code.between(*w["validation"]).to_numpy())
                                       for w in temporal["primary_windows"]])
        primary_rows.sort()
        cat, lgb = [], []
        for window in temporal["primary_windows"]:
            cutoff = window["fit"][1]
            valid = np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())
            offsets = np.searchsorted(predictions[cutoff]["rows"], valid)
            cat.extend(np.clip(predictions[cutoff]["prediction"][offsets], 0, 100))
            lgb.extend(base_predictions[cutoff]["lgb"][np.searchsorted(base_predictions[cutoff]["rows"], valid)])
        caterr = np.asarray(cat) - frame.iloc[primary_rows].threat_rank.to_numpy(float)
        lgberr = np.asarray(lgb) - frame.iloc[primary_rows].threat_rank.to_numpy(float)
        corr = float(np.corrcoef(caterr, lgberr)[0, 1])
        cat_win = float(np.mean(np.abs(caterr) < np.abs(lgberr) - 1e-10))
        lgb_win = float(np.mean(np.abs(lgberr) < np.abs(caterr) - 1e-10))
        report["catboost_lgb_complementarity"] = {
            "primary_rows": len(primary_rows), "signed_error_correlation": corr,
            "catboost_lower_absolute_error_rate": cat_win,
            "lgb_lower_absolute_error_rate": lgb_win,
            "tie_rate": 1 - cat_win - lgb_win,
        }
        eligible = corr <= 0.93 and cat_win >= 0.40
        report["fixed_catboost25_blend"] = {"evaluated": eligible,
                                             "eligibility": {"correlation_le_0_93": corr <= 0.93,
                                                             "catboost_win_rate_ge_0_40": cat_win >= 0.40}}
        if eligible:
            # This one fixed, predeclared share is evaluated only after the full six-window diagnostics.
            blend_predictions = {}
            for cutoff, fold in predictions.items():
                base = base_predictions[cutoff]
                if not np.array_equal(fold["rows"], base["rows"]):
                    raise RuntimeError(f"blend row mismatch at {cutoff}")
                blend_predictions[cutoff] = np.clip(
                    0.75 * base["lgb"] + 0.25 * np.clip(fold["prediction"], 0, 100), 0, 100
                )
            report["fixed_catboost25_blend"]["primary_pooled"] = _aggregate_external(
                frame, temporal, predictions, base_predictions, blend_predictions, "primary"
            )
            report["fixed_catboost25_blend"]["secondary_pooled"] = _aggregate_external(
                frame, temporal, predictions, base_predictions, blend_predictions, "secondary"
            )

    cache_manifest = {}
    for cutoff in sorted(predictions):
        to_cycle = int(frame.iloc[predictions[cutoff]["rows"]].cycle_code.max())
        p = CACHE / f"catboost__fit{cutoff}__to{to_cycle}.npz"
        cache_manifest[str(cutoff)] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
    report["prediction_caches"] = cache_manifest
    report["fit_budget"] = {"maximum": 6, "used": len(predictions),
                             "fit_durations_seconds": {k: v["seconds"] for k, v in metas.items()},
                             "model_binaries_saved": False}
    write_json(REPORT_DIR / "stage2_catboost.json", report)
    write_markdown(report, REPORT_DIR / "STAGE2_CATBOOST.md")
    write_figure(report, REPORT_DIR / "stage2_catboost_windows.png")
    print("pilot_gate", "continue=" + str(continue_full), "mean_delta", round(mean_delta, 6),
          "max_delta", round(max_delta, 6), "fits", len(predictions), flush=True)
    if continue_full:
        print("full primary", round(report["primary_pooled"]["candidate"]["clipped"]["mae"], 6),
              "H12", round(report["secondary_pooled"]["candidate"]["clipped"]["mae"], 6),
              "cat_lgb_err_corr", round(report["catboost_lgb_complementarity"]["signed_error_correlation"], 4),
              "blend_eval", report["fixed_catboost25_blend"]["evaluated"], flush=True)


def _aggregate_external(frame: pd.DataFrame, temporal: dict, predictions: dict,
                        base_predictions: dict, blends: dict, kind: str) -> dict:
    windows = temporal["primary_windows"] if kind == "primary" else temporal["secondary_windows"]
    rows, preds = [], []
    for window in windows:
        cutoff = window["fit"][1]
        valid = np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())
        offsets = np.searchsorted(predictions[cutoff]["rows"], valid)
        rows.append(valid)
        preds.append(blends[cutoff][offsets])
    rows, preds = np.concatenate(rows), np.concatenate(preds)
    order = np.argsort(rows)
    rows, preds = rows[order], preds[order]
    return {"n": len(rows), "candidate": metric(frame.iloc[rows], preds)}


def write_figure(report: dict, path: Path) -> None:
    pairs = list(report["windows"]["primary"].values()) + list(report["windows"]["secondary"].values())
    labels = [v["window_id"] for v in pairs]
    cat = [v["candidate"]["clipped"]["mae"] for v in pairs]
    ridge = [v["baseline_ridge25"]["clipped"]["mae"] for v in pairs]
    lgb = [v["baseline_lgb"]["clipped"]["mae"] for v in pairs]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, np.asarray(cat) - np.asarray(ridge), marker="o", linewidth=2,
            color="#2f855a", label="CatBoost − Ridge25 blend")
    ax.plot(x, np.asarray(cat) - np.asarray(lgb), marker="s", linewidth=1.8,
            color="#2b6cb0", label="CatBoost − LightGBM")
    ax.axhline(0, color="#4a5568", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Clipped MAE delta (negative is better)")
    ax.set_title("CatBoost SVD32 pilot / temporal validation")
    ax.grid(alpha=.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Stage 2 — CatBoost MAE + structured/SVD32", "",
             "## Protocol and source boundary", "",
             f"- Frozen candidate: `{report['experiment_id']}`; CatBoost {report['catboost_version']}.",
             "- Fixed CPU recipe: MAE, 500 iterations, depth 6, learning rate .05, seed 42, two threads. Native categorical columns; structured parser/imputer, TF-IDF and SVD32 are learned on each temporal fit partition.",
             "- Pilot cutoffs 55 and 67 were evaluated first against frozen LightGBM and Ridge25 predictions. The predeclared gate uses clipped MAE across W1/W4/H12_E/H12_L; full six runs only when mean delta vs Ridge25 <= +0.50 and no view > +1.00.",
             "- No Test access or external data. The cached predictions are derived Train validation vectors; no model binaries are saved.", "",
             "## Pilot gate", "",
             f"Decision: **{'continue with all six' if report['pilot_gate']['continue_full_six'] else 'retire after two pilots'}**; mean delta={report['pilot_gate']['mean_delta']:+.6f}, max delta={report['pilot_gate']['max_delta']:+.6f}.",
             "", "| View | CatBoost clipped MAE | Ridge25 MAE | Delta |", "|---|---:|---:|---:|"]
    for row in report["pilot_gate"]["views"]:
        lines.append(f"| {row['view']} | {row['catboost_clipped_mae']:.6f} | {row['ridge25_mae']:.6f} | {row['delta']:+.6f} |")
    if report["pilot_gate"]["continue_full_six"]:
        p, h = report["primary_pooled"], report["secondary_pooled"]
        lines += ["", "## Full six-window results", "",
                  f"Primary MAE raw/clipped: {p['candidate']['raw']['mae']:.6f}/{p['candidate']['clipped']['mae']:.6f}; equal-cycle MAE={p['candidate']['equal_cycle_mae']:.6f}.",
                  f"H12 pooled MAE raw/clipped: {h['candidate']['raw']['mae']:.6f}/{h['candidate']['clipped']['mae']:.6f}; H12_E={report['windows']['secondary']['H12_E']['candidate']['clipped']['mae']:.6f}; H12_L={report['windows']['secondary']['H12_L']['candidate']['clipped']['mae']:.6f}.",
                  "", "| Window | CatBoost clipped MAE | LGB clipped MAE | Ridge25 clipped MAE | Δ vs Ridge25 |", "|---|---:|---:|---:|---:|"]
        for window in list(report["windows"]["primary"].values()) + list(report["windows"]["secondary"].values()):
            c = window["candidate"]["clipped"]["mae"]
            r = window["baseline_ridge25"]["clipped"]["mae"]
            l = window["baseline_lgb"]["clipped"]["mae"]
            lines.append(f"| {window['window_id']} / n={window['n']:,} | {c:.6f} | {l:.6f} | {r:.6f} | {c-r:+.6f} |")
        comp = report["catboost_lgb_complementarity"]
        lines += ["", f"CatBoost/LGB signed-error correlation={comp['signed_error_correlation']:.4f}; CatBoost wins {comp['catboost_lower_absolute_error_rate']:.1%} of paired rows, LGB {comp['lgb_lower_absolute_error_rate']:.1%}.",
                  f"Fixed 25% CatBoost blend evaluated: {report['fixed_catboost25_blend']['evaluated']}."]
    lines += ["", "Figure: `stage2_catboost_windows.png` shows relative clipped MAE for the available pilot/full windows.",
              "", "## Interpretation", "",
              "All folds reuse Train cycles previously used in the campaign. These results are adaptive development evidence, not an untouched estimate. Review raw/clipped metrics, each window and subgroup in `stage2_catboost.json`; no Kaggle submission or model promotion is performed here.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
