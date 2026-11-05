"""Independent post-fit audit for CLS simple present-threshold D1/D2.

This audit is intentionally outside the producer's frozen source inventory.
It independently rereads only selected official Train labels and non-target
OOF cache members, then rebuilds the exact threshold optimum, predictions,
confusions, subgroup metrics, pooled gates, and ledger accounting.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/optimization_v3/cls_simple_present_threshold_v1"
RUN = ROOT / "experiments/runs/cls_simple_present_threshold_v1"
FREEZE = OUT / "SOURCE_FREEZE.json"
PREFLIGHT = OUT / "SOURCE_PREFLIGHT.json"
READY = OUT / "SOURCE_READY.json"
RELEASE = OUT / "ROOT_RELEASE_D1_D2.json"
CACHE = ROOT / "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz"
AUDIT_REPORT = OUT / "POSTFIT_INDEPENDENT_AUDIT.json"
LOW, HIGH, CENTRE = 0.45, 0.55, 0.50
MISSING_THRESHOLD = 0.53230976129477
DECISION_SHA256 = "e0e0ac69ce7ce0e148e8edfe2345e21e37addf55807cdcff6b2f2995a08cd0c5"
ROOT_ANCESTOR = "a39721172b6d293cb68ee0d120060082d222699e"
STAGES = {
    "D1": {"fit_origin": 67, "fit_cycles": (68, 71), "score_origin": 67, "score_cycles": (72, 75)},
    "D2": {"fit_origin": 67, "fit_cycles": (68, 75), "score_origin": 75, "score_cycles": (76, 79)},
}
SOURCE_PATHS = (
    "scripts/cls_simple_present_threshold_v1.py",
    "scripts/verify_cls_simple_present_threshold_v1.py",
    "tests/test_cls_simple_present_threshold_v1.py",
    "reports/optimization_v3/cls_simple_present_threshold_v1/PROTOCOL.md",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_hash(values: Any) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def metric_from_counts(n: int, class_0: int, class_1: int, tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    f1p = Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
    f1n = Fraction(2 * tn, 2 * tn + fp + fn) if 2 * tn + fp + fn else Fraction(0)
    macro = (f1p + f1n) / 2
    return {
        "n": int(n), "class_0": int(class_0), "class_1": int(class_1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "macro_f1": float(macro), "macro_f1_exact": f"{macro.numerator}/{macro.denominator}",
        "positive_precision": tp / (tp + fp) if tp + fp else 0.0,
        "positive_recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return metric_from_counts(len(y), int(np.sum(y == 0)), int(np.sum(y == 1)), tp, fp, tn, fn)


def exact_threshold_oracle(p_a: np.ndarray, missing: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Enumerate all score-induced states independently of the producer sweep."""
    p_a = np.asarray(p_a, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    y = np.asarray(y, dtype=np.int8)
    present = ~missing
    support = {str(c): int(np.sum(present & (y == c))) for c in (0, 1)}
    if min(support.values()) < 200:
        raise ValueError(f"Unsupported fit population: {support}")
    thresholds = {LOW, HIGH, CENTRE}
    in_range = np.unique(p_a[present][(p_a[present] >= LOW) & (p_a[present] <= HIGH)])
    for score in in_range:
        if LOW <= score <= HIGH:
            thresholds.add(float(score))
            immediately_above = float(np.nextafter(score, math.inf))
            if immediately_above <= HIGH:
                thresholds.add(immediately_above)
    candidates = []
    for threshold in sorted(thresholds):
        prediction = np.where(missing, p_a >= MISSING_THRESHOLD, p_a >= threshold).astype(np.int8)
        counts = metric(y, prediction)
        numerator, denominator = counts["macro_f1_exact"].split("/")
        objective = Fraction(int(numerator), int(denominator))
        candidates.append((objective, float(threshold), counts))
    best_objective = max(row[0] for row in candidates)
    tied = [row for row in candidates if row[0] == best_objective]
    selected = min(tied, key=lambda row: (abs(row[1] - CENTRE), -row[1]))
    return {
        "threshold": selected[1], "threshold_hex": selected[1].hex(),
        "fit_macro_f1": float(best_objective),
        "fit_macro_f1_exact": f"{best_objective.numerator}/{best_objective.denominator}",
        "fit_confusion": {key: selected[2][key] for key in ("tp", "fp", "tn", "fn")},
        "present_class_support": support,
        "breakpoints_considered": len(in_range),
    }


def read_train(train_path: Path):
    import pandas as pd

    wanted = ["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load"]
    columns = pd.read_csv(train_path, nrows=0).columns
    if not set(wanted + ["artifact_signal"]).issubset(columns):
        raise ValueError("Official Train schema mismatch")
    frame = pd.read_csv(train_path, usecols=wanted, dtype={"record_id": "string", "signal_stream": "string"}, low_memory=False)
    if frame["record_id"].isna().any():
        raise ValueError("Official Train has missing record IDs")
    frame["record_id"] = frame["record_id"].astype(str)
    if frame["record_id"].duplicated().any() or len(frame) != 50_608:
        raise ValueError("Official Train IDs or row count mismatch")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(int)
    frame["entry_friction"] = pd.to_numeric(frame["entry_friction"], errors="raise").astype(int)
    frame["text_load"] = pd.to_numeric(frame["text_load"], errors="raise").astype(float)
    return frame


def read_selected_labels(train_path: Path, roster_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    import pandas as pd

    positions_by_id = {rid: pos for pos, rid in enumerate(roster_ids.tolist())}
    ids = np.asarray(ids).astype(str)
    if len(positions_by_id) != len(roster_ids) or not set(ids.tolist()).issubset(positions_by_id):
        raise ValueError("Requested fit/score IDs do not match Train")
    positions = {positions_by_id[value] for value in ids.tolist()}
    raw = train_path.read_bytes()
    physical_lines = raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)
    if physical_lines != len(roster_ids) + 1:
        raise ValueError("Physical record mapping failed before label audit")
    labels = pd.read_csv(
        train_path, usecols=["record_id", "artifact_signal"],
        dtype={"record_id": "string", "artifact_signal": "Int8"},
        skiprows=lambda line: line > 0 and (line - 1) not in positions,
        low_memory=False,
    )
    labels["record_id"] = labels["record_id"].astype(str)
    if labels["record_id"].duplicated().any() or set(labels["record_id"]) != set(ids.tolist()):
        raise ValueError("Selected-only label audit returned unexpected IDs")
    mapping = dict(zip(labels["record_id"], labels["artifact_signal"].astype(int)))
    result = np.asarray([mapping[item] for item in ids], dtype=np.int8)
    if not np.isin(result, [0, 1]).all():
        raise ValueError("Official selected labels are not binary")
    return result


def load_cache_non_target():
    result = {}
    with np.load(CACHE, allow_pickle=False) as archive:
        for origin in (67, 75):
            prefix = f"E{origin}_"
            result[origin] = {
                key: np.asarray(archive[prefix + key])
                for key in ("ids", "cycles", "missing", "pA", "pP", "group_availability",
                            "group_entry_friction", "group_text_load")
            }
    return result


def arrays_for_stage(stage: str, cache_data: dict[int, dict[str, np.ndarray]], train_ids: np.ndarray) -> dict[str, np.ndarray]:
    plan = STAGES[stage]
    fit = cache_data[plan["fit_origin"]]
    score = cache_data[plan["score_origin"]]
    fit_mask = (fit["cycles"] >= plan["fit_cycles"][0]) & (fit["cycles"] <= plan["fit_cycles"][1])
    score_mask = (score["cycles"] >= plan["score_cycles"][0]) & (score["cycles"] <= plan["score_cycles"][1])
    return {
        "fit_ids": fit["ids"][fit_mask].astype(str),
        "fit_cycles": fit["cycles"][fit_mask].astype(int),
        "fit_missing": fit["missing"][fit_mask].astype(bool),
        "fit_pA": fit["pA"][fit_mask].astype(float),
        "score_ids": score["ids"][score_mask].astype(str),
        "score_cycles": score["cycles"][score_mask].astype(int),
        "score_missing": score["missing"][score_mask].astype(bool),
        "score_pA": score["pA"][score_mask].astype(float),
        "score_pP": score["pP"][score_mask].astype(float),
        "score_groups": {
            key.removeprefix("group_"): score[key][score_mask].astype(str)
            for key in ("group_availability", "group_entry_friction", "group_text_load")
        },
        "train_ids": train_ids,
    }


def metrics_by_group(y: np.ndarray, prediction: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, Any]:
    output = {}
    for dimension, labels in groups.items():
        dimension_result = {}
        for group in sorted(set(labels.tolist())):
            mask = labels == group
            n0, n1 = int(np.sum(y[mask] == 0)), int(np.sum(y[mask] == 1))
            n = int(mask.sum())
            detail = metric(y[mask], prediction[mask])
            detail.pop("n")
            detail.pop("class_0")
            detail.pop("class_1")
            dimension_result[group] = {
                "support": {"n": n, "class_0": n0, "class_1": n1},
                "status": "SUPPORTED" if n >= 200 and min(n0, n1) >= 30 else "UNKNOWN",
                "metrics": metric(y[mask], prediction[mask]),
            }
        output[dimension] = dimension_result
    return output


def assert_equal(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"{path}: key set differs")
        for key in expected:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: list length/type differs")
        for i, (left, right) in enumerate(zip(actual, expected)):
            assert_equal(left, right, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(float(actual), float(expected), rel_tol=1e-14, abs_tol=1e-15):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def audit() -> dict[str, Any]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    ready = json.loads(READY.read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    release_hash = sha(RELEASE)
    if release.get("status") != "ROOT_RELEASED_FOR_D1_D2" or release_hash != "00a316262db48f841d279d6c7914ca488710e598065adcf3cb8be73c31c0f1f8":
        raise ValueError("Root release identity/hash mismatch")
    if release.get("allowed_stages") != ["D1", "D2"] or release.get("calibrator_fit_cap") != 2 or release.get("final_fit_authorized") is not False:
        raise ValueError("Root release exceeds the two diagnostic fits or enables final fit")
    if release.get("source_commit") != freeze.get("source_commit") or release.get("source_freeze_sha256") != sha(FREEZE):
        raise ValueError("Root release does not bind the frozen source")
    if release.get("source_preflight_sha256") != sha(PREFLIGHT) or release.get("source_ready_sha256") != sha(READY):
        raise ValueError("Root release does not bind the target-free source receipts")
    if ready.get("status") != "PASS" or preflight.get("status") != "PASS_TARGET_FREE_PREFLIGHT":
        raise ValueError("Source-ready/preflight status invalid")
    train_path = Path(freeze["inputs"]["official_train_csv"]["path_at_freeze"])
    if sha(train_path) != freeze["inputs"]["official_train_csv"]["sha256"] or sha(CACHE) != freeze["inputs"]["a02_oof_cache"]["sha256"]:
        raise ValueError("Frozen official Train or OOF cache hash changed")
    if sha(CACHE) != "25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351":
        raise ValueError("A02 OOF cache differs from authorized input")
    if sha(ROOT.parent / "tk2-simplification-audit-v1/reports/simplification_v1/CLS_SIMPLE_THRESHOLD_DECISION.md") != DECISION_SHA256:
        raise ValueError("Root decision hash mismatch")
    audit_repo = ROOT.parent / "tk2-simplification-audit-v1"
    if git(audit_repo, "status", "--porcelain") or subprocess.run(
        ["git", "-C", str(audit_repo), "merge-base", "--is-ancestor", ROOT_ANCESTOR, "HEAD"],
        capture_output=True,
    ).returncode:
        raise ValueError("Root decision checkout is dirty or required ancestor is missing")
    for relative, expected_hash in freeze["source_hashes"].items():
        if sha(ROOT / relative) != expected_hash:
            raise ValueError(f"Frozen producer source changed: {relative}")
        payload = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{freeze['source_commit']}:{relative}"],
            check=True, capture_output=True,
        ).stdout
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Source commit bytes differ: {relative}")

    train = read_train(train_path)
    train_ids = train["record_id"].to_numpy(dtype=str)
    by_id = {rid: pos for pos, rid in enumerate(train_ids.tolist())}
    cache_data = load_cache_non_target()
    ledger_path = RUN / "LEDGER.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    if len({row["event_id"] for row in ledger}) != len(ledger):
        raise ValueError("Duplicate ledger event IDs")
    fit_starts = [row for row in ledger if row.get("event") == "FIT_STARTED"]
    fit_done = [row for row in ledger if row.get("event") == "FIT_FROZEN"]
    if len(fit_starts) != 2 or len(fit_done) != 2 or {row.get("stage") for row in fit_starts} != {"D1", "D2"}:
        raise ValueError("Ledger does not show exactly two completed calibrator fits")
    if any(row.get("event") in {"FIT_UNSUPPORTED", "FIT_RECOVERED"} for row in ledger):
        raise ValueError("Unexpected unsupported or recovered fit event; audit ledger manually")
    results: dict[str, Any] = {}
    checks = []
    unknowns = []
    for stage in ("D1", "D2"):
        vectors = arrays_for_stage(stage, cache_data, train_ids)
        if set(vectors["fit_ids"].tolist()) & set(vectors["score_ids"].tolist()):
            raise ValueError(f"{stage} fit and score IDs overlap")
        if not set(vectors["fit_ids"].tolist()).issubset(by_id) or not set(vectors["score_ids"].tolist()).issubset(by_id):
            raise ValueError(f"{stage} IDs are not official Train IDs")
        fit_receipt_path = RUN / "fits" / f"{stage}.json"
        score_path = RUN / "scores" / f"{stage}_RESULTS.json"
        prediction_path = RUN / "scores" / f"{stage}_FROZEN_PREDICTIONS.json"
        fit_receipt = json.loads(fit_receipt_path.read_text(encoding="utf-8"))
        score_report = json.loads(score_path.read_text(encoding="utf-8"))
        frozen = json.loads(prediction_path.read_text(encoding="utf-8"))
        fit_event = next(row for row in fit_done if row.get("stage") == stage)
        score_event = next(row for row in ledger if row.get("stage") == stage and row.get("event") == "SCORED")
        if sha(fit_receipt_path) != fit_event.get("receipt_sha256"):
            raise ValueError(f"{stage} fit receipt is not ledger-bound")
        if sha(score_path) != score_event.get("result_sha256") or sha(prediction_path) != score_event.get("frozen_predictions_sha256"):
            raise ValueError(f"{stage} score/frozen predictions are not ledger-bound")
        if fit_receipt.get("official_train_sha256") != sha(train_path) or fit_receipt.get("oof_cache_sha256") != sha(CACHE):
            raise ValueError(f"{stage} fit input lineage mismatch")
        if fit_receipt.get("fit_ids_sha256") != ordered_id_hash(vectors["fit_ids"]):
            raise ValueError(f"{stage} fit row IDs mismatch")
        if fit_receipt.get("fit_rows") != len(vectors["fit_ids"]):
            raise ValueError(f"{stage} fit row denominator mismatch")
        expected_cycle_counts = {
            str(cycle): int(np.sum(vectors["fit_cycles"] == cycle))
            for cycle in sorted(np.unique(vectors["fit_cycles"]).tolist())
        }
        assert_equal(fit_receipt["fit_cycle_counts"], expected_cycle_counts, f"{stage}.fit_cycle_counts")
        if score_report.get("frozen_predictions_sha256") != sha(prediction_path):
            raise ValueError(f"{stage} score report does not bind frozen predictions")

        fit_y = read_selected_labels(train_path, train_ids, vectors["fit_ids"])
        optimum = exact_threshold_oracle(vectors["fit_pA"], vectors["fit_missing"], fit_y)
        for key in ("threshold_hex", "fit_macro_f1_exact", "fit_confusion", "present_class_support"):
            assert_equal(fit_receipt[key], optimum[key], f"{stage}.fit.{key}")
        assert_equal(fit_receipt["threshold"], optimum["threshold"], f"{stage}.fit.threshold")
        assert_equal(fit_receipt["fit_macro_f1"], optimum["fit_macro_f1"], f"{stage}.fit.fit_macro_f1")
        assert_equal(fit_receipt["breakpoints_considered"], optimum["breakpoints_considered"], f"{stage}.fit.breakpoints")
        if fit_receipt.get("cache_target_member_opened") is not False:
            raise AssertionError(f"{stage} source receipt claims cached targets were opened")
        checks.extend([f"{stage}_fit_exact_optimum", f"{stage}_fit_confusion", f"{stage}_fit_support"])

        score_y = read_selected_labels(train_path, train_ids, vectors["score_ids"])
        expected_ids_hash = ordered_id_hash(vectors["score_ids"])
        if frozen.get("score_ids_sha256") != expected_ids_hash or score_report.get("score_ids_sha256") != expected_ids_hash:
            raise ValueError(f"{stage} score IDs differ from OOF/Train row mapping")
        if frozen.get("ids") != vectors["score_ids"].tolist() or frozen.get("cycles") != vectors["score_cycles"].tolist():
            raise ValueError(f"{stage} frozen prediction order/cycles mismatch")
        if not np.array_equal(np.asarray(frozen["missing"], dtype=bool), vectors["score_missing"]):
            raise ValueError(f"{stage} frozen missing mask mismatch")
        if not np.allclose(np.asarray(frozen["pA"], float), vectors["score_pA"], rtol=0, atol=1e-15):
            raise ValueError(f"{stage} frozen pA scores differ from cached OOF")
        if not np.allclose(np.asarray(frozen["pP"], float), vectors["score_pP"], rtol=0, atol=1e-15):
            raise ValueError(f"{stage} frozen pP scores differ from cached OOF")
        threshold = float.fromhex(fit_receipt["threshold_hex"])
        assert_equal(score_report["fit"]["threshold"], fit_receipt["threshold"], f"{stage}.score_fit_threshold")
        assert_equal(score_report["fit"]["fit_macro_f1_exact"], fit_receipt["fit_macro_f1_exact"], f"{stage}.score_fit_objective")
        if score_report["fit"]["threshold_at_interval_bound"] != (threshold in {LOW, HIGH}):
            raise AssertionError(f"{stage} threshold bound-hit flag mismatch")
        missing = vectors["score_missing"]
        pred_candidate = np.where(missing, vectors["score_pA"] >= MISSING_THRESHOLD,
                                  vectors["score_pA"] >= threshold).astype(np.int8)
        pred_simple = np.where(missing, vectors["score_pA"] >= MISSING_THRESHOLD,
                               vectors["score_pA"] >= CENTRE).astype(np.int8)
        pred_complex = np.where(missing, vectors["score_pA"] >= MISSING_THRESHOLD,
                                (0.5 * vectors["score_pP"] + 0.5 * vectors["score_pA"]) >= CENTRE).astype(np.int8)
        for key, expected_prediction in (("candidate", pred_candidate), ("simple_fixed_050", pred_simple), ("complex_fixed_050", pred_complex)):
            if not np.array_equal(np.asarray(frozen[key], dtype=np.int8), expected_prediction):
                raise ValueError(f"{stage} frozen {key} predictions do not reproduce")
        if not np.array_equal(pred_candidate[missing], pred_simple[missing]):
            raise ValueError(f"{stage} missing decisions changed")

        expected_metrics = {
            "full": {
                "candidate": metric(score_y, pred_candidate),
                "simple_fixed_050": metric(score_y, pred_simple),
                "complex_fixed_050": metric(score_y, pred_complex),
            },
            "present": {
                "candidate": metric(score_y[~missing], pred_candidate[~missing]),
                "simple_fixed_050": metric(score_y[~missing], pred_simple[~missing]),
                "complex_fixed_050": metric(score_y[~missing], pred_complex[~missing]),
            },
        }
        assert_equal(score_report["metrics"], expected_metrics, f"{stage}.metrics")
        by_cycle = {}
        for cycle in sorted(np.unique(vectors["score_cycles"]).tolist()):
            mask = vectors["score_cycles"] == cycle
            by_cycle[str(cycle)] = {
                "candidate": metric(score_y[mask], pred_candidate[mask]),
                "simple_fixed_050": metric(score_y[mask], pred_simple[mask]),
            }
        assert_equal(score_report["by_cycle"], by_cycle, f"{stage}.by_cycle")
        groups = vectors["score_groups"]
        expected_group_candidate = metrics_by_group(score_y, pred_candidate, groups)
        expected_group_simple = metrics_by_group(score_y, pred_simple, groups)
        assert_equal(score_report["subgroups"]["candidate"], expected_group_candidate, f"{stage}.subgroups.candidate")
        assert_equal(score_report["subgroups"]["simple_fixed_050"], expected_group_simple, f"{stage}.subgroups.simple")
        subgroup_delta = {}
        for dimension, by_label in expected_group_candidate.items():
            subgroup_delta[dimension] = {}
            for group, candidate_detail in by_label.items():
                baseline_detail = expected_group_simple[dimension][group]
                if candidate_detail["status"] != "SUPPORTED":
                    subgroup_delta[dimension][group] = {"status": "UNKNOWN", "reason": "support gate not met"}
                    unknowns.append(f"{stage}:{dimension}:{group}")
                else:
                    subgroup_delta[dimension][group] = {
                        "status": "SUPPORTED",
                        "macro_f1_delta": candidate_detail["metrics"]["macro_f1"] - baseline_detail["metrics"]["macro_f1"],
                        "positive_recall_delta": candidate_detail["metrics"]["positive_recall"] - baseline_detail["metrics"]["positive_recall"],
                    }
        assert_equal(score_report["subgroup_deltas_vs_simple_fixed_050"], subgroup_delta, f"{stage}.subgroup_deltas")
        delta = expected_metrics["full"]["candidate"]["macro_f1"] - expected_metrics["full"]["simple_fixed_050"]["macro_f1"]
        assert_equal(score_report["deltas_vs_simple_fixed_050"]["macro_f1"], delta, f"{stage}.delta")
        if score_report["changed_decisions_vs_simple_fixed_050"] != int(np.sum(pred_candidate != pred_simple)):
            raise ValueError(f"{stage} changed-decision count mismatch")
        checks.extend([f"{stage}_frozen_predictions", f"{stage}_full_present_metrics", f"{stage}_cycle_metrics", f"{stage}_subgroup_metrics"])
        results[stage] = {
            "threshold": threshold,
            "threshold_at_bound": threshold in {LOW, HIGH},
            "fit_rows": len(fit_y), "score_rows": len(score_y),
            "fit_macro_f1": optimum["fit_macro_f1"],
            "fit_macro_f1_exact": optimum["fit_macro_f1_exact"],
            "fit_support": optimum["present_class_support"],
            "score_full": expected_metrics["full"],
            "score_present": expected_metrics["present"],
            "per_window_macro_f1_delta": delta,
            "changed_decisions": int(np.sum(pred_candidate != pred_simple)),
            "missing_decisions_invariant": True,
            "subgroups": subgroup_delta,
            "complex_full_macro_f1": expected_metrics["full"]["complex_fixed_050"]["macro_f1"],
        }

    score_sets = [set(np.asarray([rid for rid, cyc in zip(cache_data[s["score_origin"]]["ids"], cache_data[s["score_origin"]]["cycles"])
                                  if s["score_cycles"][0] <= cyc <= s["score_cycles"][1]], dtype=str).tolist())
                  for s in STAGES.values()]
    if score_sets[0] & score_sets[1] or len(score_sets[0] | score_sets[1]) != 15_953:
        raise ValueError("Independent D1/D2 score-window disjointness/count mismatch")
    pooled = {}
    for view in ("full", "present"):
        pooled[view] = {}
        for model_name in ("candidate", "simple_fixed_050", "complex_fixed_050"):
            a = results["D1"][f"score_{view}"][model_name]
            b = results["D2"][f"score_{view}"][model_name]
            pooled_counts = {key: int(a[key]) + int(b[key]) for key in ("n", "class_0", "class_1", "tp", "fp", "tn", "fn")}
            pooled[view][model_name] = metric_from_counts(**pooled_counts)
    pooled_gain = pooled["full"]["candidate"]["macro_f1"] - pooled["full"]["simple_fixed_050"]["macro_f1"]
    window_gains = {stage: results[stage]["per_window_macro_f1_delta"] for stage in ("D1", "D2")}
    subgroup_rows = []
    for stage, stage_data in results.items():
        for dimension, groups in stage_data["subgroups"].items():
            for group, detail in groups.items():
                subgroup_rows.append({"stage": stage, "dimension": dimension, "group": group, **detail})
    supported = [row for row in subgroup_rows if row["status"] == "SUPPORTED"]
    subgroup_guard_pass = all(row["macro_f1_delta"] >= -0.01 and row["positive_recall_delta"] >= -0.02 for row in supported)
    qualification = "UNKNOWN" if unknowns else ("PASS" if pooled_gain >= 0.0003 and all(x >= 0 for x in window_gains.values()) and subgroup_guard_pass else "FAIL")
    exploratory = "UNKNOWN" if unknowns else ("PASS" if pooled_gain >= 0 and all(x >= -0.001 for x in window_gains.values()) and subgroup_guard_pass else "FAIL")
    thresholds = {stage: results[stage]["threshold"] for stage in ("D1", "D2")}
    producer_summary = json.loads((RUN / "SUMMARY.json").read_text(encoding="utf-8"))
    assert_equal(producer_summary["score_rows"], 15_953, "summary.score_rows")
    assert_equal(producer_summary["thresholds"], thresholds, "summary.thresholds")
    assert_equal(producer_summary["per_window_full_macro_f1_gain_vs_simple_fixed_050"], window_gains, "summary.window_gains")
    assert_equal(producer_summary["pooled_metrics"], pooled, "summary.pooled_metrics")
    assert_equal(producer_summary["pooled_full_macro_f1_gain_vs_simple_fixed_050"], pooled_gain, "summary.pooled_gain")
    expected_bound_hits = {stage: results[stage]["threshold_at_bound"] for stage in ("D1", "D2")}
    assert_equal(producer_summary["thresholds_at_bounds"], expected_bound_hits, "summary.threshold_bounds")
    assert_equal(producer_summary["threshold_absolute_difference_D1_D2"], abs(thresholds["D1"] - thresholds["D2"]), "summary.threshold_stability")
    assert_equal(producer_summary["complex_comparator_pooled_macro_f1"], pooled["full"]["complex_fixed_050"]["macro_f1"], "summary.complex_comparator")
    if not unknowns:
        assert_equal(producer_summary["qualification"], qualification, "summary.qualification")
        assert_equal(producer_summary["exploratory_review"], exploratory, "summary.exploratory")
    checks.extend(["D1_D2_score_windows_disjoint_15953", "pooled_confusion_metrics", "producer_summary_replay"])

    if len([row for row in ledger if row.get("event") == "SCORED"]) != 2:
        raise ValueError("Ledger does not contain exactly two completed score stages")
    if any(row.get("stage") is not None and row.get("stage") not in {"D1", "D2"} for row in ledger):
        raise ValueError("Ledger contains an unapproved stage")
    if any(row.get("event") in {"TEST_INFERENCE", "CSV_CREATED", "KAGGLE_UPLOAD"} for row in ledger):
        raise ValueError("Ledger contains an out-of-scope Test/CSV/Kaggle action")
    if release.get("predictor_fits") != 0 or release.get("preprocessing_fits") != 0 or release.get("Test_access") is not False or release.get("Kaggle_access") is not False:
        raise ValueError("Root release boundaries are inconsistent")
    checks.extend(["ledger_fit_cap_exactly_two", "ledger_score_cap_exactly_two", "no_Test_or_Kaggle"])
    report = {
        "schema_version": 1,
        "status": "PASS" if not unknowns else "PASS_WITH_UNKNOWN_SUBGROUPS",
        "campaign": "CLS_SIMPLE_PRESENT_THRESHOLD_V1",
        "source_commit": freeze["source_commit"],
        "receipt_head_at_audit": git(ROOT, "rev-parse", "HEAD"),
        "root_release_sha256": release_hash,
        "official_train_sha256": sha(train_path),
        "oof_cache_sha256": sha(CACHE),
        "ledger_sha256": sha(ledger_path),
        "fit_cap": {"authorized": 2, "started": len(fit_starts), "completed": len(fit_done)},
        "score_rows_total": 15_953,
        "stages": results,
        "pooled_metrics": pooled,
        "pooled_full_macro_f1_gain_vs_simple_fixed_050": pooled_gain,
        "per_window_full_macro_f1_gain_vs_simple_fixed_050": window_gains,
        "threshold_stability": {
            "absolute_D1_D2_difference": abs(thresholds["D1"] - thresholds["D2"]),
            "D1_at_bound": results["D1"]["threshold_at_bound"],
            "D2_at_bound": results["D2"]["threshold_at_bound"],
        },
        "subgroup_gate": {
            "supported_group_count": len(supported),
            "unknown_group_count": len(unknowns),
            "unknown_groups": unknowns,
            "all_supported_groups_within_harm_limits": subgroup_guard_pass,
            "details": subgroup_rows,
        },
        "qualification": qualification,
        "exploratory_review": exploratory,
        "secondary_complex_comparator_pooled_macro_f1": pooled["full"]["complex_fixed_050"]["macro_f1"],
        "checks": checks,
        "check_count": len(checks),
        "target_labels_read_only_for_frozen_fit_and_score_ids": True,
        "oof_target_members_opened": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_contacted": False,
    }
    return report


def main() -> int:
    report = audit()
    if AUDIT_REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite independent audit: {AUDIT_REPORT}")
    AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    temp = AUDIT_REPORT.with_name("." + AUDIT_REPORT.name + ".tmp")
    with temp.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    try:
        import os
        os.link(temp, AUDIT_REPORT)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
