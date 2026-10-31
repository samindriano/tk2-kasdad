"""Independent audit of the frozen availability-specialist experiment."""
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

from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files

from scripts.research_autonomous_classification import (
    CANDIDATES,
    FIT_ENDS,
    FREEZE_PATH,
    MODEL_NAMES,
    PROTOCOL_PATH,
    REPORT_DIR,
    ROUND1_AUDITS,
    ROUND1_CACHE,
    ROUND2_RESULTS,
    RUN_DIR,
    VOLUME_LABELS,
    expected_code_hashes,
    json_sha256,
    sha256_file,
    read_json,
    write_json,
)


def _close(actual: float | None, expected: float | None, *, label: str, atol: float = 1e-12) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0):
        raise AssertionError(f"{label}: {actual:.16g} != {expected:.16g}")


def independent_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Invalid aligned binary labels/probabilities")
    predicted = (p >= 0.50).astype(np.int8)
    labels = [0, 1]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y, predicted, labels=labels)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {"label": label, "support": int(support[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(labels)
        ],
        "positive_rate": float(y.mean()) if len(y) else None,
        "positive_prediction_rate": float(predicted.mean()) if len(y) else None,
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
    }


def compare_metrics(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if int(actual["n"]) != int(expected["n"]):
        raise AssertionError(f"{label}: n mismatch {actual['n']} != {expected['n']}")
    _close(actual["macro_f1"], expected["macro_f1"], label=f"{label}.macro_f1")
    _close(actual["positive_rate"], expected["positive_rate"], label=f"{label}.positive_rate")
    _close(actual["positive_prediction_rate"], expected["positive_prediction_rate"],
           label=f"{label}.positive_prediction_rate")
    if actual["confusion_matrix"] != expected["confusion_matrix"]:
        raise AssertionError(f"{label}: confusion matrix differs")
    if actual["false_positive"] != expected["false_positive"] or actual["false_negative"] != expected["false_negative"]:
        raise AssertionError(f"{label}: error counts differ")
    if len(actual["per_class"]) != 2:
        raise AssertionError(f"{label}: per-class metrics missing")
    for i, (actual_class, expected_class) in enumerate(zip(actual["per_class"], expected["per_class"])):
        if int(actual_class["support"]) != int(expected_class["support"]):
            raise AssertionError(f"{label}: class-{i} support differs")
        for metric in ("precision", "recall", "f1"):
            _close(actual_class[metric], expected_class[metric], label=f"{label}.class_{i}.{metric}")


def _probability_lookup(rows: np.ndarray, probability: np.ndarray) -> dict[int, float]:
    if len(rows) != len(probability) or len(np.unique(rows)) != len(rows):
        raise AssertionError("Branch rows and prediction vector are not one-to-one")
    return {int(row): float(value) for row, value in zip(rows, probability)}


def _check_branch_fit(frame: pd.DataFrame, y: np.ndarray, fit_end: int, state: dict[str, Any],
                      branch: str, expected_model_id: str, expected_mask: np.ndarray) -> dict[str, Any]:
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    fit_rows = np.flatnonzero(cycle <= fit_end)
    X_all = frame.drop(columns=["artifact_signal"])
    X_fit = X_all.iloc[fit_rows].reset_index(drop=True)
    y_fit = y[fit_rows]
    missing = stream_missing_mask(X_fit)
    mask = expected_mask(missing)
    positions = np.flatnonzero(mask)
    branch_y = y_fit[mask]
    counts = np.bincount(branch_y, minlength=2)
    ids_sha = hashlib.sha256("\n".join(X_fit.iloc[positions]["record_id"].astype(str)).encode("utf-8")).hexdigest()
    record = state["new_fits"][branch]
    expert_state = record["expert_state"]
    if record["model_id"] != expected_model_id:
        raise AssertionError(f"fit{fit_end}/{branch}: model id mismatch")
    if record["fit_rows"] != len(positions) or record["class_counts"] != counts.astype(int).tolist():
        raise AssertionError(f"fit{fit_end}/{branch}: fit support mismatch")
    if record["fit_cycle_range"] != [int(X_fit.iloc[positions]["cycle_code"].min()),
                                    int(X_fit.iloc[positions]["cycle_code"].max())]:
        raise AssertionError(f"fit{fit_end}/{branch}: selected cycle range mismatch")
    if record["fit_record_ids_sha256"] != ids_sha:
        raise AssertionError(f"fit{fit_end}/{branch}: record-id subset fingerprint mismatch")
    if expert_state["input_feature_names"] != [name for name in expert_state["input_feature_names"]
                                                 if name not in {"record_id", "artifact_signal", "threat_rank"}]:
        raise AssertionError(f"fit{fit_end}/{branch}: target or ID appeared in feature allowlist")
    model = expected_model_id
    if expert_state["id"] != model or expert_state["fit_rows"] != len(positions):
        raise AssertionError(f"fit{fit_end}/{branch}: fitted model identity/row count differs")
    fields = expert_state["input_feature_names"]
    expected_fingerprint = _fit_fingerprint(X_fit.iloc[positions].loc[:, fields], tuple(fields), branch_y)
    if expert_state["fit_data_sha256"] != expected_fingerprint or record["fit_data_sha256"] != expected_fingerprint:
        raise AssertionError(f"fit{fit_end}/{branch}: fitted data fingerprint differs")
    if record["model_state_sha256"] != expert_state["model_state_sha256"]:
        raise AssertionError(f"fit{fit_end}/{branch}: model state fingerprint mismatch")
    if not state["prediction_did_not_change_fitted_state"]["C2_B_present_only" if branch == "C2_B_present_only" else "C1_B_missing_only"]:
        raise AssertionError(f"fit{fit_end}/{branch}: predict_proba changed audit state")
    return {"n": int(len(positions)), "classes": counts.astype(int).tolist(),
            "fit_data_sha256": expected_fingerprint, "record_ids_sha256": ids_sha}


def main() -> None:
    train_file = task_files("classification")["train"]
    if train_file is None:
        raise FileNotFoundError("Official Classification Train.csv was not found")
    train_file = train_file.resolve()
    if not FREEZE_PATH.exists():
        raise FileNotFoundError("Missing frozen protocol")
    frozen = read_json(FREEZE_PATH)
    unsigned = {key: value for key, value in frozen.items() if key != "freeze_signature"}
    if frozen["freeze_signature"] != json_sha256(unsigned):
        raise AssertionError("Freeze signature failed")
    if frozen["protocol_sha256"] != sha256_file(PROTOCOL_PATH):
        raise AssertionError("Frozen protocol bytes changed")
    if frozen["source_code_sha256"] != expected_code_hashes():
        raise AssertionError("One or more frozen source files changed")
    if frozen["train"]["sha256"] != sha256_file(train_file):
        raise AssertionError("Official Train bytes changed")

    frame = pd.read_csv(train_file)
    y = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    missing_all = stream_missing_mask(frame)
    results = read_json(REPORT_DIR / "results.json")
    if results["freeze_signature"] != frozen["freeze_signature"]:
        raise AssertionError("Result freeze signature mismatch")
    if results["new_branch_fit_count"] != 12 or results["official_test_values_read"] or results["kaggle_submissions"] != 0:
        raise AssertionError("Experiment completion boundary or fit count differs")
    protocol = read_json(PROTOCOL_PATH)
    windows_by_family = {"primary": protocol["validation"]["primary_windows"],
                         "h12": protocol["validation"]["secondary_windows"]}
    fold_data: dict[int, dict[str, Any]] = {}
    branch_checks = {}
    prediction_checks = {}
    for fit_end in FIT_ENDS:
        cache_path = RUN_DIR / f"fit{fit_end}.npz"
        audit_path = RUN_DIR / f"fit{fit_end}_audit.json"
        if not cache_path.exists() or not audit_path.exists():
            raise FileNotFoundError(f"Missing fit cache/audit at cutoff {fit_end}")
        audit = read_json(audit_path)
        if sha256_file(cache_path) != audit["cache_sha256"]:
            raise AssertionError(f"fit{fit_end}: cache SHA mismatch")
        with np.load(cache_path, allow_pickle=False) as archive:
            rows = archive["rows"].astype(np.int64, copy=True)
            c1_all = archive["C1_B_all_probability"].astype(float, copy=True)
            c2_all = archive["C2_B_all_probability"].astype(float, copy=True)
            c2_rows = archive["C2_B_present_rows"].astype(np.int64, copy=True)
            c2_p = archive["C2_B_present_probability"].astype(float, copy=True)
            c1_rows = archive["C1_B_missing_rows"].astype(np.int64, copy=True)
            c1_p = archive["C1_B_missing_probability"].astype(float, copy=True)
            stored = {name: archive[f"{name}_probability"].astype(float, copy=True) for name in MODEL_NAMES}
        h12 = next(w["validation"][1] for w in protocol["validation"]["secondary_windows"]
                   if w["fit"][1] == fit_end) if fit_end in {55, 67} else None
        if h12 is not None:
            max_cycle = int(h12)
        else:
            max_cycle = max(w["validation"][1] for w in protocol["validation"]["primary_windows"]
                            if w["fit"][1] == fit_end)
        expected_rows = np.flatnonzero((cycle > fit_end) & (cycle <= max_cycle))
        if not np.array_equal(rows, expected_rows):
            raise AssertionError(f"fit{fit_end}: outer validation row IDs/cycles differ")
        missing = missing_all[rows]
        if not np.array_equal(c2_rows, rows[~missing]) or not np.array_equal(c1_rows, rows[missing]):
            raise AssertionError(f"fit{fit_end}: specialized branch prediction rows do not match availability")
        if c1_all.shape != rows.shape or c2_all.shape != rows.shape:
            raise AssertionError(f"fit{fit_end}: historical baseline vector shape mismatch")
        for model_id, vector in (("C1_B", c1_all), ("C2_B", c2_all)):
            old_cache = ROUND1_CACHE / f"fit{fit_end}__{model_id}.npz"
            old_audit = read_json(ROUND1_AUDITS / f"fit{fit_end}.json")[model_id]
            with np.load(old_cache, allow_pickle=False) as archive:
                old_rows = archive["outer_rows"].astype(np.int64, copy=True)
                old_p = archive["probability"].astype(float, copy=True)
            if not np.array_equal(old_rows, rows) or not np.array_equal(old_p, vector):
                raise AssertionError(f"fit{fit_end}: cached {model_id} vector differs from source")
            if array_sha(vector) != old_audit["probability_sha256"]:
                raise AssertionError(f"fit{fit_end}: source {model_id} hash differs from historical audit")

        present_lookup = _probability_lookup(c2_rows, c2_p)
        missing_lookup = _probability_lookup(c1_rows, c1_p)
        c2_full = np.asarray([present_lookup.get(int(row), np.nan) for row in rows])
        c1_full = np.asarray([missing_lookup.get(int(row), np.nan) for row in rows])
        reconstructed = {
            "C4_HGB": np.where(missing, c1_all, c2_all),
            "present_specialist": np.where(missing, c1_all, c2_full),
            "missing_specialist": np.where(missing, c1_full, c2_all),
            "both_specialist": np.where(missing, c1_full, c2_full),
        }
        for name, vector in reconstructed.items():
            if not np.isfinite(vector).all() or not np.array_equal(vector, stored[name]):
                raise AssertionError(f"fit{fit_end}: route reconstruction differs for {name}")
            if array_sha(vector) != audit["route_prediction_sha256"][name]:
                raise AssertionError(f"fit{fit_end}: route hash differs for {name}")
        if array_sha(c2_p) != audit["branch_prediction_sha256"]["C2_B_present_only"]:
            raise AssertionError(f"fit{fit_end}: present branch hash mismatch")
        if array_sha(c1_p) != audit["branch_prediction_sha256"]["C1_B_missing_only"]:
            raise AssertionError(f"fit{fit_end}: missing branch hash mismatch")
        branch_checks[str(fit_end)] = {
            "present": _check_branch_fit(frame, y, fit_end, audit, "C2_B_present_only", "C2_B",
                                          lambda mask: ~mask),
            "missing": _check_branch_fit(frame, y, fit_end, audit, "C1_B_missing_only", "C1_B",
                                          lambda mask: mask),
        }
        prediction_checks[str(fit_end)] = {
            "rows": int(len(rows)), "c4_sha256": array_sha(reconstructed["C4_HGB"]),
            "present_only_sha256": array_sha(reconstructed["present_specialist"]),
            "missing_only_sha256": array_sha(reconstructed["missing_specialist"]),
            "both_sha256": array_sha(reconstructed["both_specialist"]),
        }
        fold_data[fit_end] = {"rows": rows, **reconstructed}

    recomputed = independent_evaluation(frame, y, fold_data, windows_by_family)
    for family in ("primary", "h12"):
        for name in MODEL_NAMES:
            recorded = results["aggregate"][family][name]
            compare_metrics(recorded["metrics"], recomputed[family][name]["metrics"], f"{family}/{name}")
            _close(recorded["equal_cycle_macro_f1"], recomputed[family][name]["equal_cycle_macro_f1"],
                   label=f"{family}/{name}.equal_cycle")
            for group, actual in recomputed[family][name]["subgroups"].items():
                compare_metrics(recorded["subgroups"][group], actual, f"{family}/{name}/{group}")

    round2 = read_json(ROUND2_RESULTS)["aggregate"]
    for family, old_family in (("primary", "primary"), ("h12", "secondary")):
        recorded = results["aggregate"][family]["C4_HGB"]
        expected = round2[old_family]["fixed_0.5"]["C4_HGB"]
        _close(recorded["metrics"]["macro_f1"], expected["metrics"]["macro_f1"], label=f"C4/{family}/Round2")
        _close(recorded["equal_cycle_macro_f1"], expected["equal_cycle_macro_f1"], label=f"C4/{family}/equal-cycle")

    report = {
        "status": "PASS",
        "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["train"]["sha256"],
        "source_code_hashes_match": True,
        "historical_controls_match_original_caches_and_audits": True,
        "new_fits": 12,
        "fit_subset_and_target_boundary_checks": branch_checks,
        "route_prediction_hash_checks": prediction_checks,
        "independent_sklearn_metric_reconciliation": {
            family: {name: {
                "n": recomputed[family][name]["metrics"]["n"],
                "macro_f1": recomputed[family][name]["metrics"]["macro_f1"],
                "equal_cycle_macro_f1": recomputed[family][name]["equal_cycle_macro_f1"],
                "subgroups_checked": sorted(recomputed[family][name]["subgroups"]),
            } for name in MODEL_NAMES} for family in ("primary", "h12")
        },
        "official_test_values_read": False,
        "official_test_target_used": False,
        "kaggle_submissions": 0,
    }
    write_json(REPORT_DIR / "verification.json", report)
    print(json.dumps({"status": "PASS", "new_fits": 12,
                      "primary": {name: results["summary"][name]["primary_macro_f1"] for name in MODEL_NAMES},
                      "h12": {name: results["summary"][name]["h12_macro_f1"] for name in MODEL_NAMES},
                      "report": str(REPORT_DIR / "verification.json")}, indent=2))


def array_sha(array: np.ndarray) -> str:
    values = np.asarray(array, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def independent_evaluation(frame: pd.DataFrame, y: np.ndarray, fold_data: dict[int, dict[str, Any]],
                           windows_by_family: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    missing = stream_missing_mask(frame)
    volume = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=float)
    out: dict[str, Any] = {}
    for family, windows in windows_by_family.items():
        model_y = {name: [] for name in MODEL_NAMES}
        model_p = {name: [] for name in MODEL_NAMES}
        model_rows = {name: [] for name in MODEL_NAMES}
        model_bins = {name: [] for name in MODEL_NAMES}
        for window in windows:
            fit_end = int(window["fit"][1])
            lo, hi = map(int, window["validation"])
            fold = fold_data[fit_end]
            rows = fold["rows"]
            mask = (cycle[rows] >= lo) & (cycle[rows] <= hi)
            selected_rows = rows[mask]
            quantiles = np.quantile(volume[cycle <= fit_end], [0.2, 0.4, 0.6, 0.8])
            bins = np.searchsorted(quantiles, volume[selected_rows], side="right")
            for name in MODEL_NAMES:
                model_y[name].append(y[selected_rows])
                model_p[name].append(fold[name][mask])
                model_rows[name].append(selected_rows)
                model_bins[name].append(bins)
        out[family] = {}
        for name in MODEL_NAMES:
            yy = np.concatenate(model_y[name])
            pp = np.concatenate(model_p[name])
            rr = np.concatenate(model_rows[name])
            per_cycle = []
            for cycle_value in np.unique(cycle[rr]):
                selected = cycle[rr] == cycle_value
                per_cycle.append(independent_metrics(yy[selected], pp[selected])["macro_f1"])
            groups = {}
            for group, mask in (("stream_present", ~missing[rr]), ("stream_missing", missing[rr])):
                groups[group] = independent_metrics(yy[mask], pp[mask])
            bin_all = np.concatenate(model_bins[name])
            for index, label in enumerate(VOLUME_LABELS):
                mask = bin_all == index
                groups[f"text_load_{label}"] = independent_metrics(yy[mask], pp[mask])
            out[family][name] = {
                "metrics": independent_metrics(yy, pp),
                "equal_cycle_macro_f1": float(np.mean(per_cycle)),
                "subgroups": groups,
            }
    return out


if __name__ == "__main__":
    main()
