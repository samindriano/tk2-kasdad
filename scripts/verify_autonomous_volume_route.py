"""Independent route, boundary, and sklearn metric verification for q80 route."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files

from scripts.research_autonomous_volume_route import (
    CODE_FILES,
    FIT_ENDS,
    FREEZE_PATH,
    H12,
    MODELS,
    PRIMARY,
    PROTOCOL_PATH,
    REPORT_DIR,
    ROUND1_AUDITS,
    ROUND1_RUN,
    RUN_DIR,
    SOURCE_EXPERIMENT,
    SOURCE_REPORT,
    SOURCE_RUN,
    VOLUME_LABELS,
    array_sha256,
    code_hashes,
    file_sha256,
    json_sha256,
    read_json,
    write_json,
)


def independent_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Expected aligned binary target and finite [0,1] probabilities")
    pred = (p >= 0.50).astype(np.int8)
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=labels)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {"label": k, "support": int(support[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i])}
            for i, k in enumerate(labels)
        ],
        "positive_rate": float(y.mean()) if len(y) else None,
        "positive_prediction_rate": float(pred.mean()) if len(y) else None,
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
    }


def assert_close(a: float | None, b: float | None, label: str, atol: float = 1e-12) -> None:
    if a is None or b is None:
        if a is not b:
            raise AssertionError(f"{label}: {a!r} != {b!r}")
    elif not np.isclose(float(a), float(b), atol=atol, rtol=0):
        raise AssertionError(f"{label}: {a:.16g} != {b:.16g}")


def assert_metric_record(recorded: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    if recorded["n"] != actual["n"] or recorded["confusion_matrix"] != actual["confusion_matrix"]:
        raise AssertionError(f"{label}: denominator/confusion matrix mismatch")
    for key in ("macro_f1", "positive_rate", "positive_prediction_rate"):
        assert_close(recorded[key], actual[key], f"{label}.{key}")
    if recorded["false_positive"] != actual["false_positive"] or recorded["false_negative"] != actual["false_negative"]:
        raise AssertionError(f"{label}: false positive/negative mismatch")
    for index in (0, 1):
        left, right = recorded["per_class"][index], actual["per_class"][index]
        if left["support"] != right["support"]:
            raise AssertionError(f"{label}: class-{index} support mismatch")
        for key in ("precision", "recall", "f1"):
            assert_close(left[key], right[key], f"{label}.class_{index}.{key}")


def _vector_lookup(rows: np.ndarray, probability: np.ndarray) -> dict[int, float]:
    if len(rows) != len(probability) or len(np.unique(rows)) != len(rows):
        raise AssertionError("Specialist rows/probabilities do not align uniquely")
    return {int(row): float(p) for row, p in zip(rows, probability)}


def _read_source_fold(fit_end: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = SOURCE_RUN / f"fit{fit_end}.npz"
    audit_path = SOURCE_RUN / f"fit{fit_end}_audit.json"
    audit = read_json(audit_path)
    if file_sha256(path) != audit["cache_sha256"]:
        raise AssertionError(f"Source run cache hash mismatch at {fit_end}")
    with np.load(path, allow_pickle=False) as z:
        vectors = {key: z[key].copy() for key in [
            "rows", "C1_B_all_probability", "C2_B_all_probability", "C2_B_present_rows",
            "C2_B_present_probability"]}
    return vectors, audit


def _load_prediction(fit_end: int, expected_sha: str) -> tuple[np.ndarray, np.ndarray]:
    path = RUN_DIR / f"fit{fit_end}.npz"
    if not path.exists() or file_sha256(path) != expected_sha:
        raise AssertionError(f"Volume-route prediction cache hash mismatch at fit{fit_end}")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"rows", "probability"}:
            raise AssertionError(f"Unexpected cache contents at fit{fit_end}")
        return z["rows"].astype(np.int64), z["probability"].astype(float)


def independent_recompute(frame: pd.DataFrame, y: np.ndarray,
                          folds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    volume = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=float)
    missing = stream_missing_mask(frame)
    out: dict[str, Any] = {}
    windows_out: list[dict[str, Any]] = []
    for family, windows in (("primary", PRIMARY), ("h12", H12)):
        out[family] = {}
        model_y = {name: [] for name in MODELS}
        model_p = {name: [] for name in MODELS}
        model_rows = {name: [] for name in MODELS}
        model_bins = {name: [] for name in MODELS}
        model_high = {name: [] for name in MODELS}
        for window in windows:
            end = int(window["fit"][1])
            lo, hi = map(int, window["validation"])
            fold = folds[end]
            rows = fold["rows"]
            select = (cycle[rows] >= lo) & (cycle[rows] <= hi)
            selected_rows = rows[select]
            yy = y[selected_rows]
            qq = np.quantile(volume[cycle <= end], [0.2, 0.4, 0.6, 0.8])
            bins = np.searchsorted(qq, volume[selected_rows], side="right")
            high = (~missing[selected_rows]) & (volume[selected_rows] > qq[-1])
            for name in MODELS:
                pp = fold[name][select]
                metric = independent_metrics(yy, pp)
                windows_out.append({"family": family, "window": window["id"], "model": name,
                                    "n": len(selected_rows), "metrics": metric})
                model_y[name].append(yy)
                model_p[name].append(pp)
                model_rows[name].append(selected_rows)
                model_bins[name].append(bins)
                model_high[name].append(high)
        for name in MODELS:
            yy = np.concatenate(model_y[name])
            pp = np.concatenate(model_p[name])
            rows = np.concatenate(model_rows[name])
            bins = np.concatenate(model_bins[name])
            high = np.concatenate(model_high[name])
            cycle_scores = [independent_metrics(yy[cycle[rows] == value], pp[cycle[rows] == value])["macro_f1"]
                            for value in np.unique(cycle[rows])]
            masks = {
                "stream_missing": missing[rows],
                "present_not_above_q80": (~missing[rows]) & ~high,
                "present_above_q80": high,
                "all_rows_q80_volume_bin": bins == 4,
            }
            for index, label in enumerate(VOLUME_LABELS):
                masks[f"text_load_{label}"] = bins == index
            out[family][name] = {
                "metrics": independent_metrics(yy, pp),
                "equal_cycle_macro_f1": float(np.mean(cycle_scores)),
                "subgroups": {group: independent_metrics(yy[mask], pp[mask]) for group, mask in masks.items()},
            }
    return {"aggregate": out, "windows": windows_out}


def main() -> None:
    source = task_files("classification")["train"]
    if source is None:
        raise FileNotFoundError("Official Classification Train.csv is missing")
    source = source.resolve()
    frame = pd.read_csv(source)
    if file_sha256(source) != read_json(FREEZE_PATH)["official_train"]["sha256"]:
        raise AssertionError("Official Train source hash differs")
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(target, [0, 1]).all():
        raise AssertionError("Target is not binary")
    frozen = read_json(FREEZE_PATH)
    if json_sha256({k: v for k, v in frozen.items() if k != "freeze_signature"}) != frozen["freeze_signature"]:
        raise AssertionError("Volume route freeze signature invalid")
    if frozen["protocol_sha256"] != file_sha256(PROTOCOL_PATH):
        raise AssertionError("Frozen volume route protocol changed")
    if frozen["source_code_sha256"] != code_hashes():
        raise AssertionError("Frozen route source code changed")
    for relative, expected in frozen["read_only_cache_files_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise AssertionError(f"Read-only source cache/report changed: {relative}")
    results = read_json(REPORT_DIR / "results.json")
    if results["new_temporal_model_fits"] != 0 or results["official_test_values_read"] or results["kaggle_submissions"] != 0:
        raise AssertionError("Unexpected model-fit/Test/submission boundary")
    if results["freeze_signature"] != frozen["freeze_signature"]:
        raise AssertionError("Results use a different freeze")

    missing = stream_missing_mask(frame)
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    volume = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=float)
    folds: dict[int, dict[str, Any]] = {}
    route_checks = {}
    for fit_end in FIT_ENDS:
        source_vectors, source_audit = _read_source_fold(fit_end)
        rows = source_vectors["rows"].astype(np.int64)
        output_rows, route_p = _load_prediction(fit_end, results["prediction_cache_sha256"][str(fit_end)])
        if not np.array_equal(output_rows, rows):
            raise AssertionError(f"fit{fit_end}: output row order differs from source vectors")
        q80 = float(frozen["folds"][str(fit_end)]["q80"])
        train_q80 = float(np.quantile(volume[cycle <= fit_end], 0.80))
        assert_close(q80, train_q80, f"fit{fit_end}.q80")
        row_missing = missing[rows]
        val_high = (~row_missing) & (volume[rows] > q80)
        present_lookup = _vector_lookup(source_vectors["C2_B_present_rows"],
                                        source_vectors["C2_B_present_probability"])
        specialized = np.asarray([present_lookup.get(int(row), np.nan) for row in rows])
        if not np.isfinite(specialized[~row_missing]).all() or np.isfinite(specialized[row_missing]).any():
            raise AssertionError(f"fit{fit_end}: specialized branch predictions do not map to present rows")
        c1 = source_vectors["C1_B_all_probability"].astype(float)
        c2 = source_vectors["C2_B_all_probability"].astype(float)
        c4 = np.where(row_missing, c1, c2)
        uniform = np.where(row_missing, c1, specialized)
        expected = np.where(row_missing, c1, np.where(val_high, specialized, c2))
        if not np.isfinite(route_p).all() or not np.array_equal(route_p, expected):
            raise AssertionError(f"fit{fit_end}: q80 output differs from exact branch composition")
        if not np.array_equal(route_p[~val_high], c4[~val_high]):
            raise AssertionError(f"fit{fit_end}: missing/non-high rows changed from C4")
        if not np.array_equal(route_p[val_high], uniform[val_high]):
            raise AssertionError(f"fit{fit_end}: high-volume rows differ from uniform expert")
        folds[fit_end] = {"rows": rows, "C4_HGB": c4, "present_specialist": uniform,
                          "present_q80_route": route_p, "high_present": val_high}
        route_checks[str(fit_end)] = {
            "rows": int(len(rows)),
            "q80": q80,
            "high_present_rows": int(val_high.sum()),
            "unchanged_missing_and_present_not_high": True,
            "high_rows_equal_uniform_present_specialist": True,
            "probability_sha256": array_sha256(route_p),
        }

    recomputed = independent_recompute(frame, target, folds)
    for family in ("primary", "h12"):
        for name in MODELS:
            recorded = results["aggregate"][family][name]
            actual = recomputed["aggregate"][family][name]
            assert_metric_record(recorded["metrics"], actual["metrics"], f"{family}/{name}")
            assert_close(recorded["equal_cycle_macro_f1"], actual["equal_cycle_macro_f1"],
                         f"{family}/{name}.equal_cycle")
            for group, subgroup_metric in actual["subgroups"].items():
                assert_metric_record(recorded["subgroups"][group], subgroup_metric,
                                     f"{family}/{name}/{group}")
    recorded_windows = results["windows"]
    if len(recorded_windows) != len(recomputed["windows"]):
        raise AssertionError("Window metrics row count differs")
    for recorded, actual in zip(recorded_windows, recomputed["windows"]):
        if (recorded["family"], recorded["window"], recorded["model"]) != \
           (actual["family"], actual["window"], actual["model"]):
            raise AssertionError("Window metric identity/order differs")
        assert_metric_record(recorded, actual["metrics"],
                             f"window/{actual['family']}/{actual['window']}/{actual['model']}")

    report = {
        "status": "PASS",
        "freeze_signature": frozen["freeze_signature"],
        "source_development_freeze_signature": frozen["source_development_experiment"]["freeze_signature"],
        "input_sha256": frozen["official_train"]["sha256"],
        "source_caches_read_only_and_unchanged": True,
        "new_temporal_model_fits": 0,
        "route_composition_identity_checks": route_checks,
        "independent_sklearn_metrics": {
            family: {name: {
                "n": recomputed["aggregate"][family][name]["metrics"]["n"],
                "macro_f1": recomputed["aggregate"][family][name]["metrics"]["macro_f1"],
                "equal_cycle_macro_f1": recomputed["aggregate"][family][name]["equal_cycle_macro_f1"],
                "subgroups": sorted(recomputed["aggregate"][family][name]["subgroups"]),
            } for name in MODELS} for family in ("primary", "h12")
        },
        "official_test_values_read": False,
        "kaggle_submissions": 0,
    }
    write_json(REPORT_DIR / "verification.json", report)
    print(json.dumps({"status": "PASS", "new_temporal_model_fits": 0,
                      "primary": {name: results["summary"][name]["primary_macro_f1"] for name in MODELS},
                      "h12": {name: results["summary"][name]["h12_macro_f1"] for name in MODELS},
                      "report": str(REPORT_DIR / "verification.json")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
