"""Independent metric replay R2 for the frozen missing-offset pilot."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "verify_reg_missing_median_offset_v1.py"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / "REG_MISSING_MEDIAN_OFFSET_V1"
SCORE_REPORT = REPORT_ROOT / "SCORE_R1.json"
PRELABEL_QA = REPORT_ROOT / "PRELABEL_QA_R1.json"
OUTPUT = REPORT_ROOT / "METRIC_QA_R2.json"
MIN_SUPPORT = 100
EXPECTED_ROWS = 15953


class MetricQAError(RuntimeError):
    pass


def _load_base():
    spec = importlib.util.spec_from_file_location("missing_offset_r1_source_verifier", BASE_PATH)
    if spec is None or spec.loader is None:
        raise MetricQAError("Cannot load frozen R1 source verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.float64)
    pred = np.asarray(p, dtype=np.float64)
    residual = target - pred
    error = pred - target
    absolute = np.abs(error)
    return {
        "rows": int(target.size),
        "mae": float(absolute.mean()),
        "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
        "median_residual_y_minus_prediction": float(np.median(residual)),
        "mean_residual_y_minus_prediction": float(residual.mean()),
        "bias_prediction_minus_target": float(error.mean()),
    }


def _masks(target: np.ndarray, prediction: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = np.asarray(prediction["missing"], dtype=bool)
    impact = np.asarray(prediction["impact_quartile"], dtype=np.int8)
    text = np.asarray(prediction["text_quartile"], dtype=np.int8)
    friction = np.asarray(prediction["entry_friction"], dtype=np.float64)
    masks = {
        "missing": missing,
        "present": ~missing,
        "target_lt5": target < 5.0,
        "target_ge95": target >= 95.0,
        "EF0": friction == 0.0,
        "EF1": friction > 0.0,
    }
    for index in range(4):
        masks[f"impact_load_Q{index + 1}"] = impact == index
        masks[f"text_load_Q{index + 1}"] = text == index
    return masks


def _groups(target: np.ndarray, champion: np.ndarray, candidate: np.ndarray,
            masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw_mask in masks.items():
        mask = np.asarray(raw_mask, dtype=bool)
        count = int(mask.sum())
        if count < MIN_SUPPORT:
            result[name] = {"status": "UNKNOWN", "rows": count, "champion": None,
                            "candidate": None, "harm_candidate_minus_champion": None}
        else:
            before = _metric(target[mask], champion[mask])
            after = _metric(target[mask], candidate[mask])
            result[name] = {"status": "MEASURED", "rows": count, "champion": before,
                            "candidate": after,
                            "harm_candidate_minus_champion": after["mae"] - before["mae"]}
    return result


def _fold_metrics(name: str, pred: Mapping[str, np.ndarray], target: np.ndarray,
                  base: Any) -> dict[str, Any]:
    missing = np.asarray(pred["missing"], dtype=bool)
    champion = np.asarray(pred["champion_hinge"], dtype=np.float64)
    candidate = np.asarray(pred["candidate_hinge"], dtype=np.float64)
    if not np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8")):
        raise MetricQAError(f"Present champion bits changed in {name}")
    if len(target) != len(champion) or not np.isfinite(target).all():
        raise MetricQAError(f"Invalid score target vector in {name}")

    n_missing = int(missing.sum())
    before_missing = _metric(target[missing], champion[missing])
    after_missing = _metric(target[missing], candidate[missing])
    if n_missing < MIN_SUPPORT:
        missing_report = {"rows": n_missing, "status": "UNKNOWN", "champion": None,
                          "candidate": None, "mae_gain_candidate_vs_champion": None,
                          "mae_harm_candidate_minus_champion": None}
    else:
        missing_report = {"rows": n_missing, "status": "MEASURED", "champion": before_missing,
                          "candidate": after_missing,
                          "mae_gain_candidate_vs_champion": before_missing["mae"] - after_missing["mae"],
                          "mae_harm_candidate_minus_champion": after_missing["mae"] - before_missing["mae"]}

    cycles: dict[str, Any] = {}
    cycle_values = np.asarray(pred["cycles"], dtype=np.int64)
    for cycle in np.unique(cycle_values):
        cmask = cycle_values == cycle
        mmask = cmask & missing
        n = int(mmask.sum())
        cycle_before = _metric(target[cmask], champion[cmask])
        cycle_after = _metric(target[cmask], candidate[cmask])
        cycles[str(int(cycle))] = {
            "rows": int(cmask.sum()), "missing_rows": n,
            "missing_status": "MEASURED" if n >= MIN_SUPPORT else "UNKNOWN",
            "champion": cycle_before, "candidate": cycle_after,
            "overall_mae_gain_candidate_vs_champion": cycle_before["mae"] - cycle_after["mae"],
            "missing_champion": _metric(target[mmask], champion[mmask]) if n >= MIN_SUPPORT else None,
            "missing_candidate": _metric(target[mmask], candidate[mmask]) if n >= MIN_SUPPORT else None,
            "missing_mae_gain_candidate_vs_champion": (
                _metric(target[mmask], champion[mmask])["mae"] -
                _metric(target[mmask], candidate[mmask])["mae"] if n >= MIN_SUPPORT else None),
        }
    before = _metric(target, champion)
    after = _metric(target, candidate)
    identity = base._canonical_sha({
        "rows": base._sha_array(pred["query_rows"]),
        "ids": base._sha_array(pred["query_ids"]),
        "cycles": base._sha_array(pred["cycles"]),
    })
    fold_masks = _masks(target, pred)
    return {
        "origin": 67 if name == "D1" else 75,
        "score_cycles": [72, 75] if name == "D1" else [76, 79],
        "rows": len(target),
        "missing": missing_report,
        "overall": {"champion": before, "candidate": after,
                    "mae_gain_candidate_vs_champion": before["mae"] - after["mae"]},
        "per_cycle": cycles,
        "fixed_groups": _groups(target, champion, candidate, fold_masks),
        "present_bit_exact": True,
        "row_identity_sha256": identity,
        "score_labels_read": int(len(target)),
    }


def run() -> dict[str, Any]:
    base = _load_base()
    source, _ = base._source()
    pre = base._read_json(PRELABEL_QA)
    base._verify_signed(pre, "qa_signature", "prelabel QA")
    if pre.get("status") != "PASS" or pre.get("score_labels_read") is not False:
        raise MetricQAError("Independent metric replay blocked before prelabel PASS")
    score = base._read_json(SCORE_REPORT)
    base._verify_signed(score, "score_signature", "score report")
    if (score.get("status") != "SCORED" or score.get("source_signature") != source.get("source_signature")
            or score.get("prelabel_qa_sha256") != base._sha_file(PRELABEL_QA)
            or score.get("independent_metric_qa_pass") is not False):
        raise MetricQAError("Score report/source/prelabel binding failed")

    folds: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, np.ndarray] = {}
    for name in ("D1", "D2"):
        _state, _receipt, pred = base._load_output(name)
        target = base._read_targets(pred["query_rows"], pred["query_ids"].astype(str))
        fold = _fold_metrics(name, pred, target, base)
        if not base._same_tree(fold, score["folds"][name]):
            raise MetricQAError(f"Independent fold metric replay differs: {name}")
        folds[name] = fold
        predictions[name] = pred
        targets[name] = target

    pooled_y = np.concatenate([targets[name] for name in ("D1", "D2")])
    pooled_rows = np.concatenate([predictions[name]["query_rows"] for name in ("D1", "D2")])
    pooled_ids = np.concatenate([predictions[name]["query_ids"] for name in ("D1", "D2")])
    pooled_ids_string = pooled_ids.astype(str)
    pooled_base = np.concatenate([predictions[name]["champion_hinge"] for name in ("D1", "D2")]).astype(np.float64)
    pooled_candidate = np.concatenate([predictions[name]["candidate_hinge"] for name in ("D1", "D2")]).astype(np.float64)
    pooled_missing = np.concatenate([predictions[name]["missing"] for name in ("D1", "D2")]).astype(bool)
    pooled_meta = {key: np.concatenate([predictions[name][key] for name in ("D1", "D2")])
                   for key in ("entry_friction", "impact_quartile", "text_quartile")}
    overall_before = _metric(pooled_y, pooled_base)
    overall_after = _metric(pooled_y, pooled_candidate)
    overall = {"champion": overall_before, "candidate": overall_after,
               "mae_gain_candidate_vs_champion": overall_before["mae"] - overall_after["mae"]}
    miss_before = _metric(pooled_y[pooled_missing], pooled_base[pooled_missing])
    miss_after = _metric(pooled_y[pooled_missing], pooled_candidate[pooled_missing])
    missing = {"rows": int(pooled_missing.sum()), "status": "MEASURED" if int(pooled_missing.sum()) >= MIN_SUPPORT else "UNKNOWN",
               "champion": miss_before if int(pooled_missing.sum()) >= MIN_SUPPORT else None,
               "candidate": miss_after if int(pooled_missing.sum()) >= MIN_SUPPORT else None,
               "mae_gain_candidate_vs_champion": miss_before["mae"] - miss_after["mae"] if int(pooled_missing.sum()) >= MIN_SUPPORT else None,
               "mae_harm_candidate_minus_champion": miss_after["mae"] - miss_before["mae"] if int(pooled_missing.sum()) >= MIN_SUPPORT else None}
    pooled_pred = {"missing": pooled_missing, **pooled_meta}
    group_metrics = _groups(pooled_y, pooled_base, pooled_candidate,
                            _masks(pooled_y, pooled_pred))
    pooled_identity = base._canonical_sha({"rows": base._sha_array(pooled_rows),
                                           "ids": base._sha_array(pooled_ids_string)})
    if (len(pooled_y) != EXPECTED_ROWS or len(np.unique(pooled_rows)) != EXPECTED_ROWS
            or len(np.unique(pooled_ids_string)) != EXPECTED_ROWS
            or score.get("pooled_rows") != EXPECTED_ROWS
            or score.get("pooled_row_identity_sha256") != pooled_identity
            or not base._same_tree(overall, score["pooled_overall"])
            or not base._same_tree(missing, score["pooled_missing"])
            or not base._same_tree(group_metrics, score["pooled_fixed_groups"])):
        raise MetricQAError("Independent pooled metric, identity, or group replay differs")

    harms = [value["harm_candidate_minus_champion"] for value in group_metrics.values()
             if value["status"] == "MEASURED"]
    maximum_harm = max(harms, default=None)
    both_supported = all(folds[name]["missing"]["rows"] >= MIN_SUPPORT for name in ("D1", "D2"))
    present_exact = all(folds[name]["present_bit_exact"] for name in ("D1", "D2"))
    pooled_gain = overall["mae_gain_candidate_vs_champion"]
    group_pass = maximum_harm is not None and maximum_harm <= 0.25
    scientific = bool(both_supported
                      and all(folds[name]["missing"]["mae_gain_candidate_vs_champion"] > 0
                              for name in ("D1", "D2"))
                      and pooled_gain > 0 and group_pass and present_exact)
    exploratory = bool(both_supported and pooled_gain > 0
                       and all(folds[name]["missing"]["mae_harm_candidate_minus_champion"] <= 0.10
                               for name in ("D1", "D2"))
                       and group_pass and present_exact)
    if (score.get("maximum_supported_group_harm") != maximum_harm
            or score.get("both_fold_missing_support_pass") != both_supported
            or score.get("present_predictions_bit_exact") != present_exact
            or score.get("scientific_gate_without_independent_metric_qa") != scientific
            or score.get("exploratory_gate_without_independent_metric_qa") != exploratory):
        raise MetricQAError("Independent gate reconstruction differs")

    body = {
        "schema_version": 1,
        "experiment_id": "REG_MISSING_MEDIAN_OFFSET_V1",
        "status": "PASS",
        "source_signature": source["source_signature"],
        "score_signature": score["score_signature"],
        "score_sha256": base._sha_file(SCORE_REPORT),
        "prelabel_qa_sha256": base._sha_file(PRELABEL_QA),
        "metric_verifier": "R2_ADDENDUM",
        "metric_verifier_sha256": base._sha_file(Path(__file__)),
        "R1_verifier_failure": "API mismatch in pooled fixed-group mask assembly; failed before writing a metric QA report",
        "independent_metric_replay": "PASS",
        "independent_target_rows_read": len(pooled_y),
        "target_values_persisted": False,
        "test_or_sample_accessed": False,
        "pooled_overall": overall,
        "pooled_missing": missing,
        "pooled_fixed_groups": group_metrics,
        "D1": folds["D1"],
        "D2": folds["D2"],
        "pooled_mae_gain": pooled_gain,
        "D1_missing_gain": folds["D1"]["missing"]["mae_gain_candidate_vs_champion"],
        "D2_missing_gain": folds["D2"]["missing"]["mae_gain_candidate_vs_champion"],
        "maximum_supported_group_harm": maximum_harm,
        "present_predictions_bit_exact": present_exact,
        "scientific_qualification": scientific,
        "exploratory_benchmark_eligible": exploratory,
    }
    result = dict(body)
    result["qa_signature"] = base._canonical_sha(result)
    payload = base._canonical(result) + b"\n"
    if OUTPUT.exists() and OUTPUT.read_bytes() != payload:
        raise MetricQAError("Refusing to overwrite changed metric QA R2")
    if not OUTPUT.exists():
        OUTPUT.write_bytes(payload)
    return {"status": result["status"], "qa_signature": result["qa_signature"],
            "pooled_mae_gain": pooled_gain, "D1_missing_gain": result["D1_missing_gain"],
            "D2_missing_gain": result["D2_missing_gain"],
            "maximum_supported_group_harm": maximum_harm,
            "scientific_qualification": scientific,
            "exploratory_benchmark_eligible": exploratory}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), sort_keys=True, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise SystemExit(1)
