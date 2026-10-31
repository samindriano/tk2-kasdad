"""Reconcile paired historical regression predictions and test small blends.

This stage reuses verified caches only, reads official Regression Train only,
and never opens official Test or performs a new model fit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.campaign_regression import WEIGHT_GRID, blend_predictions, select_blend_weight
from src.models.regression_research_common import mapped_scores
from src.models.regression_round2_evaluation import load_protocol, positions_digest
from src.models.regression_round2_metrics import paired, summarize
from src.models.round1_evaluation import array_digest, digest_json
from src.utils.paths import task_files

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments/dual_campaign/regression/protocol.json"
FREEZE_PATH = ROOT / "experiments/dual_campaign/regression/freeze.json"
REPORT_DIR = ROOT / "reports/dual_campaign/regression"
CACHE = ROOT / "experiments/runs/dual_reg_v1"
HIST_CACHE = ROOT / "experiments/runs/regression_research_v1"
HIST_REPORT = ROOT / "reports/regression_research"
HIST_CONFIG = ROOT / "experiments/regression_research"
FOLD_ENDS = (55, 59, 63, 67, 71, 75)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               allow_nan=False, indent=2) + "\n", encoding="utf-8")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def assert_positions(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.dtype.kind not in "iu" or not np.array_equal(actual.astype(np.int64), expected):
        raise RuntimeError(f"cached source row positions differ: {label}")


def read_inner(path: Path, fit: np.ndarray, valid: np.ndarray,
               param: str, label: str) -> np.ndarray:
    meta = read_json(path.with_suffix(".json"))
    arrays = read_npz(path)
    assert_positions(arrays["fit_rows"], fit, label + " fit")
    assert_positions(arrays["validation_rows"], valid, label + " validation")
    pred = arrays["prediction"].astype(np.float64, copy=False)
    if pred.shape != (len(valid),) or not np.isfinite(pred).all():
        raise RuntimeError(f"invalid inner prediction vector: {label}")
    if meta.get("param") != param or meta.get("fit_rows") != len(fit):
        raise RuntimeError(f"inner recipe/fit row mismatch: {label}")
    if meta.get("fit_positions_sha256") != positions_digest(fit):
        raise RuntimeError(f"inner fit position digest mismatch: {label}")
    if meta.get("validation_positions_sha256") != positions_digest(valid):
        raise RuntimeError(f"inner validation position digest mismatch: {label}")
    if meta.get("prediction_sha256") != array_digest(pred):
        raise RuntimeError(f"inner prediction digest mismatch: {label}")
    if meta.get("state_sha256") != digest_json(meta.get("state")):
        raise RuntimeError(f"inner model state digest mismatch: {label}")
    if meta.get("validation_labels_passed_to_fit") is not False:
        raise RuntimeError(f"inner validation fit guard missing: {label}")
    if meta.get("test_features_used_for_fit") not in (False, None):
        raise RuntimeError(f"Test boundary failed: {label}")
    return pred


def read_outer(raw_path: Path, mapped_path: Path, rows: np.ndarray,
               cycle: np.ndarray, expected_raw_sha: str, weight: float,
               label: str) -> np.ndarray:
    raw_arrays, mapped_arrays = read_npz(raw_path), read_npz(mapped_path)
    assert_positions(raw_arrays["validation_rows"], rows, label + " raw rows")
    assert_positions(mapped_arrays["validation_rows"], rows, label + " mapped rows")
    raw = raw_arrays["prediction"].astype(np.float64, copy=False)
    mapped = mapped_arrays["prediction"].astype(np.float64, copy=False)
    if raw.shape != (len(rows),) or not np.isfinite(raw).all():
        raise RuntimeError(f"invalid raw outer predictions: {label}")
    if array_digest(raw) != expected_raw_sha:
        raise RuntimeError(f"outer prediction hash mismatch: {label}")
    if not np.array_equal(mapped, mapped_scores(raw, cycle[rows], weight)):
        raise RuntimeError(f"outer rank mapping mismatch: {label}")
    return mapped


def lookup(report: dict, cutoff: int, recipe: str | None = None) -> dict:
    for item in report["inner"]:
        if item.get("fit_end") == cutoff and (recipe is None or item.get("recipe") == recipe):
            return item["selected"]
    raise RuntimeError(f"missing prior-only selection at cutoff {cutoff}: {recipe}")


def rows_for(frame: pd.DataFrame, window: dict) -> np.ndarray:
    return np.flatnonzero(frame.cycle_code.between(*window["validation"]).to_numpy())


def metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict:
    return summarize(frame.reset_index(drop=True), pred, include_impact=False)


def paired_metrics(frame: pd.DataFrame, candidate: np.ndarray, control: np.ndarray) -> dict:
    return paired(frame.reset_index(drop=True), candidate, control)


def main() -> None:
    protocol = read_json(PROTOCOL_PATH)
    temporal = load_protocol()
    train_path = task_files("regression")["train"]
    if train_path is None:
        raise FileNotFoundError("official Regression Train.csv is missing")
    train_hash = file_sha256(train_path)
    lgb_freeze = read_json(HIST_CONFIG / "stage5_lgb_text_freeze.json")
    ridge_freeze = read_json(HIST_CONFIG / "stage23_models_freeze.json")
    lgb_report = read_json(HIST_REPORT / "stage5_lgb_text.json")
    ridge_report = read_json(HIST_REPORT / "stage23_models.json")
    if train_hash != lgb_freeze["identity"]["regression_input_sha256"]["train"]:
        raise RuntimeError("official Regression Train is not the frozen historical source")
    if train_hash != ridge_freeze["identity"]["regression_input_sha256"]["train"]:
        raise RuntimeError("Hybrid Ridge and LightGBM did not use identical Train bytes")
    for freeze, label in ((lgb_freeze, "LightGBM"), (ridge_freeze, "Hybrid Ridge")):
        for rel, expected in freeze["identity"]["code_sha256"].items():
            if file_sha256(ROOT / rel) != expected:
                raise RuntimeError(f"{label} source-code hash differs from freeze: {rel}")

    frame = pd.read_csv(train_path)
    cycle = frame.cycle_code.to_numpy()
    y = frame.threat_rank.to_numpy(dtype=np.float64)
    if len(frame) != 50608 or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise RuntimeError("Train shape or target domain differs from the verified source")

    folds: dict[int, dict] = {}
    selections: dict[str, dict] = {}
    for cutoff in FOLD_ENDS:
        inner_fit = np.flatnonzero(frame.cycle_code.between(47, cutoff - 4).to_numpy())
        inner_valid = np.flatnonzero(frame.cycle_code.between(cutoff - 3, cutoff).to_numpy())
        last_valid = max(w["validation"][1] for w in temporal["primary_windows"] + temporal["secondary_windows"]
                         if w["fit"][1] == cutoff)
        outer_rows = np.flatnonzero(frame.cycle_code.between(cutoff + 1, last_valid).to_numpy())
        if not (cycle[inner_fit].max() < cycle[inner_valid].min() <= cutoff < cycle[outer_rows].min()):
            raise RuntimeError(f"temporal boundaries failed at fit_end={cutoff}")

        lgb_choice = lookup(lgb_report, cutoff)
        ridge_choice = lookup(ridge_report, cutoff, "hybrid_ridge")
        lgb_inner_raw = read_inner(
            HIST_CACHE / f"lgbtext__{lgb_choice['param']}__fit{cutoff-4}__to{cutoff}.npz",
            inner_fit, inner_valid, lgb_choice["param"], f"LightGBM inner {cutoff}",
        )
        ridge_inner_raw = read_inner(
            HIST_CACHE / f"model__hybrid_ridge__base__fit{cutoff-4}__to{cutoff}.npz",
            inner_fit, inner_valid, "base", f"Ridge inner {cutoff}",
        )
        lgb_inner = mapped_scores(lgb_inner_raw, cycle[inner_valid], lgb_choice["rank_weight"])
        ridge_inner = mapped_scores(ridge_inner_raw, cycle[inner_valid], ridge_choice["rank_weight"])
        weight_selection = select_blend_weight(y[inner_valid], lgb_inner, ridge_inner, weights=WEIGHT_GRID)

        lgb_outer_raw_sha = lgb_report["audits"][str(cutoff)]["outer"]["prediction_sha256"]
        ridge_outer_raw_sha = ridge_report["audits"][str(cutoff)]["hybrid_ridge"]["outer"]["prediction_sha256"]
        lgb_outer = read_outer(
            HIST_CACHE / f"eval_lgbtext__fit{cutoff}__raw.npz",
            HIST_CACHE / f"eval_lgbtext__fit{cutoff}__inner.npz",
            outer_rows, cycle, lgb_outer_raw_sha, lgb_choice["rank_weight"], f"LightGBM outer {cutoff}",
        )
        ridge_outer = read_outer(
            HIST_CACHE / f"eval__fit{cutoff}__hybrid_ridge__raw.npz",
            HIST_CACHE / f"eval__fit{cutoff}__hybrid_ridge__inner.npz",
            outer_rows, cycle, ridge_outer_raw_sha, ridge_choice["rank_weight"], f"Ridge outer {cutoff}",
        )
        folds[cutoff] = {
            "outer_rows": outer_rows, "lgb_outer": lgb_outer, "ridge_outer": ridge_outer,
            "selected_weight": weight_selection.ridge_weight,
        }
        selections[str(cutoff)] = {
            "inner_fit_rows": len(inner_fit), "inner_validation_rows": len(inner_valid),
            "inner_validation_cycles": [cutoff - 3, cutoff],
            "lgb_component_selection": lgb_choice, "ridge_component_selection": ridge_choice,
            "ridge_share": {
                "selected": weight_selection.ridge_weight,
                "inner_mae": weight_selection.inner_mae,
                "grid": [{"weight": w, "mae": value} for w, value in weight_selection.candidates],
            },
        }

    def candidate_predictions(name: str, mode: str, cutoff: int) -> np.ndarray:
        f = folds[cutoff]
        alpha = float(mode) if mode != "selected" else f["selected_weight"]
        return blend_predictions(f["lgb_outer"], f["ridge_outer"], alpha)

    definitions = {"lgb_text_inner_v1": "0", "ridge_25": "0.25", "ridge_50": "0.5",
                   "hybrid_ridge_inner_v1": "1", "inner_selected": "selected"}
    results: dict[str, dict] = {}
    for name, mode in definitions.items():
        item = {"candidate_name": name, "primary": {}, "secondary": {}}
        for kind, windows in (("primary", temporal["primary_windows"]),
                              ("secondary", temporal["secondary_windows"])):
            for window in windows:
                cutoff = window["fit"][1]
                rows = rows_for(frame, window)
                outer_rows = folds[cutoff]["outer_rows"]
                offsets = np.searchsorted(outer_rows, rows)
                if not np.array_equal(outer_rows[offsets], rows):
                    raise RuntimeError(f"window rows missing from outer cache: {window['id']}/{name}")
                pred_outer = candidate_predictions(name, mode, cutoff)
                pred = pred_outer[offsets]
                lgb_control = folds[cutoff]["lgb_outer"][offsets]
                item[kind][window["id"]] = {
                    "cutoff": cutoff, "ridge_share": (folds[cutoff]["selected_weight"] if mode == "selected" else float(mode)),
                    "n": len(rows), "candidate": metrics(frame.iloc[rows], pred),
                    "paired_vs_lgb": paired_metrics(frame.iloc[rows], pred, lgb_control),
                }
        for kind, windows in (("primary", temporal["primary_windows"]),
                              ("secondary", temporal["secondary_windows"])):
            row_parts, pred_parts, control_parts = [], [], []
            for window in windows:
                cutoff = window["fit"][1]
                rows = rows_for(frame, window)
                offsets = np.searchsorted(folds[cutoff]["outer_rows"], rows)
                row_parts.append(rows)
                pred_parts.append(candidate_predictions(name, mode, cutoff)[offsets])
                control_parts.append(folds[cutoff]["lgb_outer"][offsets])
            all_rows = np.concatenate(row_parts)
            pred_all = np.concatenate(pred_parts)
            control_all = np.concatenate(control_parts)
            order = np.argsort(all_rows)
            all_rows, pred_all, control_all = all_rows[order], pred_all[order], control_all[order]
            if len(np.unique(all_rows)) != len(all_rows):
                raise RuntimeError(f"overlapping windows in {kind}")
            item[kind + "_pooled"] = {
                "n": len(all_rows), "candidate": metrics(frame.iloc[all_rows], pred_all),
                "paired_vs_lgb": paired_metrics(frame.iloc[all_rows], pred_all, control_all),
            }
        results[name] = item

    primary_rows = np.concatenate([rows_for(frame, w) for w in temporal["primary_windows"]])
    primary_rows.sort()
    lgb_vec, ridge_vec = [], []
    for row in primary_rows:
        cutoff = next(c for c, fold in folds.items() if np.any(fold["outer_rows"] == row))
        offset = int(np.searchsorted(folds[cutoff]["outer_rows"], row))
        lgb_vec.append(folds[cutoff]["lgb_outer"][offset])
        ridge_vec.append(folds[cutoff]["ridge_outer"][offset])
    lgb_error = np.asarray(lgb_vec) - y[primary_rows]
    ridge_error = np.asarray(ridge_vec) - y[primary_rows]
    complementarity = {
        "primary_rows": len(primary_rows),
        "lgb_mae": float(np.mean(np.abs(lgb_error))),
        "ridge_mae": float(np.mean(np.abs(ridge_error))),
        "signed_error_correlation": float(np.corrcoef(lgb_error, ridge_error)[0, 1]),
        "ridge_lower_absolute_error_rate": float(np.mean(np.abs(ridge_error) < np.abs(lgb_error) - 1e-10)),
        "lgb_lower_absolute_error_rate": float(np.mean(np.abs(lgb_error) < np.abs(ridge_error) - 1e-10)),
        "absolute_error_tie_rate": float(np.mean(np.abs(np.abs(ridge_error) - np.abs(lgb_error)) <= 1e-10)),
    }
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    old_freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8")) if FREEZE_PATH.exists() else None
    if old_freeze and old_freeze.get("freeze_chronology") is None:
        initial_path = FREEZE_PATH.with_name("freeze_initial_checksum.json")
        if not initial_path.exists():
            write_json(initial_path, old_freeze)
        superseded_sha = old_freeze["freeze_sha256"]
        freeze_created = datetime.now(timezone.utc).isoformat()
    elif old_freeze:
        superseded_sha = old_freeze["freeze_chronology"]["initial_checksum_sha256"]
        freeze_created = old_freeze["freeze_chronology"]["consolidated_freeze_created_at_utc"]
    else:
        superseded_sha = None
        freeze_created = datetime.now(timezone.utc).isoformat()
    initial_score_time = "2026-10-31T08:05:26Z"
    campaign_sources = {
        "src/models/campaign_regression.py": file_sha256(ROOT / "src/models/campaign_regression.py"),
        "scripts/research_campaign_regression.py": file_sha256(Path(__file__).resolve()),
        "scripts/verify_campaign_regression.py": file_sha256(ROOT / "scripts/verify_campaign_regression.py"),
        "tests/test_campaign_regression.py": file_sha256(ROOT / "tests/test_campaign_regression.py"),
    }
    freeze = {
        "experiment_id": protocol["experiment_id"], "source_commit": source_commit,
        "protocol_sha256": file_sha256(PROTOCOL_PATH), "official_regression_train_sha256": train_hash,
        "stage5_freeze_signature": lgb_freeze["signature"], "stage23_freeze_signature": ridge_freeze["signature"],
        "stage5_report_sha256": file_sha256(HIST_REPORT / "stage5_lgb_text.json"),
        "stage23_report_sha256": file_sha256(HIST_REPORT / "stage23_models.json"),
        "campaign_source_sha256": campaign_sources,
        "new_model_fits": 0, "outer_fit_ends": list(FOLD_ENDS), "official_test_opened": False,
        "freeze_chronology": {
            "protocol_and_initial_runner_written_before_first_score": True,
            "protocol_written_at_utc": datetime.fromtimestamp(PROTOCOL_PATH.stat().st_mtime, timezone.utc).isoformat(),
            "initial_score_report_created_at_utc": initial_score_time,
            "consolidated_freeze_created_at_utc": freeze_created,
            "checksum_freeze_preceded_initial_score": False,
            "description": "Protocol and initial runner were written before scoring; checksum receipt was consolidated after the first exploratory score. Fixed alpha candidates were declared before score review; no predictive model was newly fitted.",
            "initial_checksum_sha256": superseded_sha,
        },
    }
    freeze["freeze_sha256"] = digest_json(freeze)
    if old_freeze and old_freeze.get("freeze_chronology") is not None and old_freeze != freeze:
        raise RuntimeError("experiment freeze changed; create a new protocol version")
    write_json(FREEZE_PATH, freeze)

    CACHE.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for cutoff, fold in folds.items():
        arrays[f"rows_{cutoff}"] = fold["outer_rows"]
        arrays[f"lgb_{cutoff}"] = fold["lgb_outer"]
        arrays[f"ridge_{cutoff}"] = fold["ridge_outer"]
        for label, alpha in (("ridge25", 0.25), ("ridge50", 0.5),
                             ("selected", fold["selected_weight"])):
            arrays[f"blend_{label}_{cutoff}"] = blend_predictions(
                fold["lgb_outer"], fold["ridge_outer"], alpha
            )
    cache_path = CACHE / "stage1_paired_blends.npz"
    np.savez_compressed(cache_path, **arrays)
    report = {
        "experiment_id": protocol["experiment_id"], "freeze_sha256": freeze["freeze_sha256"],
        "source_commit": source_commit, "official_train_rows": len(frame),
        "freeze_chronology": freeze["freeze_chronology"],
        "campaign_source_sha256": campaign_sources,
        "official_train_cycle_range": [int(cycle.min()), int(cycle.max())],
        "source_rows_used": {"primary": results["lgb_text_inner_v1"]["primary_pooled"]["n"],
                             "h12": results["lgb_text_inner_v1"]["secondary_pooled"]["n"]},
        "adaptive_reuse_notice": "All temporal windows have been seen in prior development. Nested past-only alpha selection does not restore an untouched holdout; H12 overlaps primary.",
        "weight_selection": selections, "component_complementarity": complementarity,
        "results": results,
        "checks": {"raw_train_sha256_matches_both_freezes": True,
                   "historical_model_code_hashes_match": True,
                   "inner_model_state_and_prediction_hashes_match": True,
                   "outer_model_prediction_hashes_match": True,
                   "outer_mapping_reproduced_exactly": True,
                   "temporal_position_boundaries_match": True,
                   "test_features_or_labels_used": False, "new_model_fits": 0},
        "derived_prediction_cache": {"path": str(cache_path.relative_to(ROOT)),
                                     "sha256": file_sha256(cache_path), "git_ignored": True},
    }
    write_json(REPORT_DIR / "stage1_paired_blend.json", report)
    write_figures(results, REPORT_DIR)
    write_summary(report, REPORT_DIR / "STAGE1_PAIRED_BLEND.md")
    for name, result in results.items():
        print(name,
              "primary", round(result["primary_pooled"]["candidate"]["raw"]["mae"], 6),
              "equal_cycle", round(result["primary_pooled"]["candidate"]["equal_cycle_mae"], 6),
              "H12", round(result["secondary_pooled"]["candidate"]["raw"]["mae"], 6),
              "H12_E", round(result["secondary"]["H12_E"]["candidate"]["raw"]["mae"], 6),
              "H12_L", round(result["secondary"]["H12_L"]["candidate"]["raw"]["mae"], 6))


def write_figures(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = list(results)
    labels = ["LGB champion", "Ridge 25%", "Ridge 50%", "Ridge champion", "Inner-selected"]
    x = np.arange(len(names))
    primary = [results[n]["primary_pooled"]["candidate"]["raw"]["mae"] for n in names]
    h12 = [results[n]["secondary_pooled"]["candidate"]["raw"]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    width = 0.36
    ax.bar(x - width / 2, primary, width, color="#2b6cb0", label="Primary pooled")
    ax.bar(x + width / 2, h12, width, color="#dd6b20", label="H12 pooled")
    ax.set_xticks(x, labels, rotation=10, ha="right")
    ax.set_ylabel("MAE (lebih rendah lebih baik)")
    ax.set_title("Error model dan blend pada temporal validation")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "stage1_pooled_mae.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    colors = {"ridge_25": "#dd6b20", "ridge_50": "#c53030",
              "inner_selected": "#2f855a", "hybrid_ridge_inner_v1": "#718096"}
    for ax, group, keys, title in [
        (axes[0], "primary", [f"W{i}" for i in range(1, 7)], "Primary windows"),
        (axes[1], "secondary", ["H12_E", "H12_L"], "H12 windows"),
    ]:
        base = np.array([results["lgb_text_inner_v1"][group][k]["candidate"]["raw"]["mae"] for k in keys])
        xx = np.arange(len(keys))
        for name, color in colors.items():
            values = np.array([results[name][group][k]["candidate"]["raw"]["mae"] for k in keys])
            ax.plot(xx, values - base, marker="o", linewidth=1.8, color=color, label=name)
        ax.axhline(0, color="#4a5568", linestyle="--", linewidth=1)
        ax.set_xticks(xx, keys)
        ax.set_title(title)
        ax.set_ylabel("MAE delta vs LGB (negative is better)")
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stage1_window_delta.png", dpi=170)
    plt.close(fig)


def write_summary(report: dict, path: Path) -> None:
    results = report["results"]
    header = "| Candidate | Primary MAE | Equal-cycle MAE | H12 pooled | H12_E | H12_L | Better primary windows vs LGB |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    lines = ["# Stage 1 — Paired LightGBM / Hybrid Ridge blend", "",
             "## Question and design",
             "Do the frozen Hybrid Ridge predictions add complementary information to the LightGBM SVD32 champion? Ridge share is α. Component rank mappings use each model's own existing past-only selected policy; the ensemble uses the predeclared fixed α values 0.25/0.50 or selects α on a strictly earlier inner window.", "",
             f"- Experiment freeze: `{report['freeze_sha256']}`; source commit `{report['source_commit']}`.",
             "- Chronology: protocol/initial runner existed before scoring; the checksum receipt was consolidated after the first exploratory result. This is not described as a full pre-score cryptographic freeze.",
             f"- Official Train: {report['official_train_rows']:,} rows, cycles {report['official_train_cycle_range'][0]}–{report['official_train_cycle_range'][1]}.",
             f"- Evaluation support: {report['source_rows_used']['primary']:,} primary rows and {report['source_rows_used']['h12']:,} H12 rows. H12 overlaps primary.",
             "- Inner α selection at each cutoff uses mapped component predictions for cycles E−3..E generated from fits ending at E−4; outer evaluation uses fits ending at E and validation strictly after E.",
             "- Prediction caches were reused. Component code, Train SHA, model-state and prediction hashes, row positions, past-only boundaries and rank mapping were checked. No Test access or new model fit.",
             "- All validation cycles were used in earlier adaptive research; results are development evidence, not an untouched estimate.", "",
             "## Results", "", header, separator]
    for name in results:
        item = results[name]
        p = item["primary_pooled"]["candidate"]
        h = item["secondary_pooled"]["candidate"]
        count = sum(item["primary"][f"W{i}"]["candidate"]["raw"]["mae"] <
                    results["lgb_text_inner_v1"]["primary"][f"W{i}"]["candidate"]["raw"]["mae"]
                    for i in range(1, 7))
        lines.append(f"| {name} | {p['raw']['mae']:.6f} | {p['equal_cycle_mae']:.6f} | {h['raw']['mae']:.6f} | {item['secondary']['H12_E']['candidate']['raw']['mae']:.6f} | {item['secondary']['H12_L']['candidate']['raw']['mae']:.6f} | {count}/6 |")
    lines += ["", "## Trade-off detail", "",
              "| Window | LightGBM | Ridge 25% | Δ vs LGB | Ridge 50% | Δ vs LGB |",
              "|---|---:|---:|---:|---:|---:|"]
    for w in [f"W{i}" for i in range(1, 7)]:
        values = {n: results[n]["primary"][w]["candidate"]["raw"]["mae"]
                  for n in ("lgb_text_inner_v1", "ridge_25", "ridge_50")}
        lines.append(f"| {w} | {values['lgb_text_inner_v1']:.6f} | {values['ridge_25']:.6f} | {values['ridge_25']-values['lgb_text_inner_v1']:+.6f} | {values['ridge_50']:.6f} | {values['ridge_50']-values['lgb_text_inner_v1']:+.6f} |")
    for w in ("H12_E", "H12_L"):
        values = {n: results[n]["secondary"][w]["candidate"]["raw"]["mae"]
                  for n in ("lgb_text_inner_v1", "ridge_25", "ridge_50")}
        lines.append(f"| {w} | {values['lgb_text_inner_v1']:.6f} | {values['ridge_25']:.6f} | {values['ridge_25']-values['lgb_text_inner_v1']:+.6f} | {values['ridge_50']:.6f} | {values['ridge_50']-values['lgb_text_inner_v1']:+.6f} |")
    lines += ["", "Subgroup MAE on pooled primary support:", "",
              "| Subgroup | n | LightGBM | Ridge 25% | Δ | Ridge 50% | Δ |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("EF0", "EF1", "tail_low_le5", "tail_high_ge95", "stream_missing", "stream_present"):
        lgb = results["lgb_text_inner_v1"]["primary_pooled"]["candidate"]["subgroups"][group]
        r25 = results["ridge_25"]["primary_pooled"]["candidate"]["subgroups"][group]
        r50 = results["ridge_50"]["primary_pooled"]["candidate"]["subgroups"][group]
        lines.append(f"| {group} | {lgb['n']:,} | {lgb['mae']:.6f} | {r25['mae']:.6f} | {r25['mae']-lgb['mae']:+.6f} | {r50['mae']:.6f} | {r50['mae']-lgb['mae']:+.6f} |")
    lines += ["", "Inner-selected Ridge shares by cutoff 55/59/63/67/71/75: " + ", ".join(str(report["weight_selection"][str(c)]["ridge_share"]["selected"]) for c in (55, 59, 63, 67, 71, 75)) + ". This adaptive policy is included as a sensitivity check and is less stable than the fixed 25% blend.", ""]
    comp = report["component_complementarity"]
    lines += ["", "## Error complementarity", "",
              f"On the {comp['primary_rows']:,} paired primary rows, signed-error correlation was {comp['signed_error_correlation']:.4f}. Ridge had lower absolute error on {comp['ridge_lower_absolute_error_rate']:.1%} of rows; LightGBM on {comp['lgb_lower_absolute_error_rate']:.1%}; ties {comp['absolute_error_tie_rate']:.1%}. This shows potential for complementarity but does not establish that a blend generalizes.", "",
              "## Interpretation limits", "",
              "Inspect per-window/per-cycle paired results and EF, tail, and stream subgroups in `stage1_paired_blend.json`. Do not promote based only on pooled MAE. Public Kaggle scores were not used in selection. Stage 2 model-family work awaits MAIN's decision.", "",
              "Figures: `stage1_pooled_mae.png` compares pooled primary/H12; `stage1_window_delta.png` compares each temporal window with LightGBM. Negative delta favors blending.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
