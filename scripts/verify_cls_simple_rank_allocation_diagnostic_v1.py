"""Independent verifier for the zero-fit POS/A02 rank-allocation diagnostic.

This module intentionally does not import the producer. It reopens only the
frozen cached E67/E75 arrays and independently rebuilds ID order, tie-resolved
top-K sets, overlaps, confusion counts, Macro F1, ROC AUC, and average precision.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / "tk2-simplification-audit-v1"
OUT = ROOT / "reports/optimization_v3/cls_simple_rank_allocation_diagnostic_v1"
FREEZE = OUT / "PRE_ACCESS_FREEZE.json"
RESULTS = OUT / "RESULTS.json"
QA = OUT / "INDEPENDENT_QA.json"
INPUTS = {
    "audit_input_freeze": "../tk2-simplification-audit-v1/reports/simplification_v1/INPUT_FREEZE.json",
    "audit_protocol": "../tk2-simplification-audit-v1/reports/simplification_v1/PROTOCOL.md",
    "audit_script": "../tk2-simplification-audit-v1/scripts/simplification_audit_v1.py",
    "audit_decisions": "../tk2-simplification-audit-v1/reports/simplification_v1/DECISIONS.json",
    "audit_results": "../tk2-simplification-audit-v1/reports/simplification_v1/RESULTS.json",
    "audit_verification": "../tk2-simplification-audit-v1/reports/simplification_v1/VERIFICATION.json",
    "audit_findings": "../tk2-simplification-audit-v1/reports/simplification_v1/FINDINGS.md",
    "light_parallel_policy": "../tk2-simplification-audit-v1/reports/simplification_v1/LIGHT_PARALLEL_EXECUTION_POLICY.json",
    "a02_policy_predictions": "experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz",
    "pos_predictions": "experiments/runs/optimization_v3_cls_pos_one_sided_ridge_v1/predictions.npz",
    "novelty_b01_summary": "reports/optimization_v3/a02_b01_centering_v1/B01_OUTPUTS/SUMMARY.md",
    "novelty_a02_neighbors": "reports/optimization_v3/a02_neighbors_v1/FINAL_SUMMARY.md",
    "novelty_a02_cgeom": "reports/optimization_v3/a02_coefficient_geometry_v1/RESULTS.md",
    "novelty_a02_continuation": "reports/optimization_v3/a02_continuation_v1/FINAL_HANDOFF.md",
}
EXPECTED_SOURCE_PATHS = (
    "scripts/cls_simple_rank_allocation_diagnostic_v1.py",
    "scripts/verify_cls_simple_rank_allocation_diagnostic_v1.py",
    "reports/optimization_v3/cls_simple_rank_allocation_diagnostic_v1/PROTOCOL.md",
)
EXPECTED_VIEWS = {
    "E67_W4": [(67, 68, 71)],
    "E67_W5": [(67, 72, 75)],
    "E67_W6": [(67, 76, 79)],
    "E67_H12": [(67, 68, 79)],
    "E75_W6": [(75, 76, 79)],
    "UNIQUE": [(67, 68, 75), (75, 76, 79)],
}
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
FROZEN_AUDIT_HEAD = "56958254e3b33a54f583343e2cdbc18cff2b9e6a"
EXPECTED_ROWS = {"E67_W4": 4_950, "E67_W5": 7_462, "E67_W6": 8_491,
                 "E67_H12": 20_903, "E75_W6": 8_491, "UNIQUE": 20_903}
POLICIES = (
    "complex_threshold", "simple_threshold", "simple_topk_cycle_at_complex_count",
    "complex_topk_cycle_at_simple_count", "simple_topk_origin_at_complex_count",
    "complex_topk_origin_at_simple_count",
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def resolve(label: str) -> Path:
    return (ROOT / INPUTS[label]).resolve()


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def ordered_hash(ids: Any) -> str:
    return hashlib.sha256("\n".join(str(x) for x in ids).encode("utf-8")).hexdigest()


def set_hash(ids: Any) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in ids)).encode("utf-8")).hexdigest()


def verify_lineage(freeze: dict[str, Any]) -> None:
    for label, relative in INPUTS.items():
        actual = sha_file(resolve(label))
        if actual != freeze["input_hashes"].get(label):
            raise ValueError(f"Input hash mismatch: {label}")
    sources = freeze["source_hashes"]
    for relative in EXPECTED_SOURCE_PATHS:
        if sha_file(ROOT / relative) != sources.get(relative):
            raise ValueError(f"Source hash mismatch: {relative}")
    if freeze.get("views") != {k: [list(v) for v in vals] for k, vals in EXPECTED_VIEWS.items()}:
        raise ValueError("Frozen view map differs from independent verifier")
    if git(AUDIT, "rev-parse", "HEAD") != FROZEN_AUDIT_HEAD or git(AUDIT, "status", "--porcelain"):
        raise ValueError("Frozen root-audit worktree changed")
    ancestor = subprocess.run(["git", "-C", str(AUDIT), "merge-base", "--is-ancestor",
                               freeze["root_audit"]["source_head"], "HEAD"], capture_output=True)
    if ancestor.returncode != 0:
        raise ValueError("Frozen source commit is not an ancestor of audit artifacts")


def load_vectors(freeze: dict[str, Any]) -> dict[int, dict[str, Any]]:
    import numpy as np

    cache = resolve("a02_policy_predictions")
    data: dict[int, dict[str, Any]] = {}
    with np.load(cache, allow_pickle=False) as z:
        for origin in (67, 75):
            prefix = f"E{origin}_"
            data[origin] = {
                "ids": np.asarray(z[prefix + "ids"]).astype(str),
                "cycles": np.asarray(z[prefix + "cycles"], dtype=np.int64),
                "missing": np.asarray(z[prefix + "missing"], dtype=bool),
                "y": np.asarray(z[prefix + "y"], dtype=np.int8),
                "pA": np.asarray(z[prefix + "pA"], dtype=np.float64),
                "pP": np.asarray(z[prefix + "pP"], dtype=np.float64),
            }
            d = data[origin]
            n = len(d["ids"])
            if any(len(d[k]) != n for k in d):
                raise ValueError(f"E{origin} array lengths mismatch")
            if len(set(d["ids"].tolist())) != n:
                raise ValueError(f"E{origin} IDs are not unique")
            if not np.isin(d["y"], [0, 1]).all() or not np.isfinite(d["pA"]).all():
                raise ValueError(f"E{origin} labels/pA invalid")
            present = ~d["missing"]
            if not np.isfinite(d["pP"][present]).all():
                raise ValueError(f"E{origin} present pP is not finite")
            if ((d["pA"] < 0) | (d["pA"] > 1)).any() or ((d["pP"][present] < 0) | (d["pP"][present] > 1)).any():
                raise ValueError(f"E{origin} probability outside [0,1]")
    return data


def make_view(data: dict[int, dict[str, Any]], segments: list[tuple[int, int, int]]) -> dict[str, Any]:
    import numpy as np

    chunks: dict[str, list[Any]] = {k: [] for k in ("ids", "origins", "cycles", "missing", "y", "pA", "pP")}
    for origin, lo, hi in segments:
        d = data[origin]
        mask = (d["cycles"] >= lo) & (d["cycles"] <= hi)
        for name in ("ids", "missing", "y", "pA", "pP"):
            chunks[name].append(d[name][mask])
        chunks["cycles"].append(d["cycles"][mask])
        chunks["origins"].append(np.repeat(np.int16(origin), int(mask.sum())))
    view = {k: np.concatenate(v) for k, v in chunks.items()}
    if len(set(view["ids"].tolist())) != len(view["ids"]):
        raise ValueError("Repeated record_id in view")
    return view


def independent_topk(ix: list[int], scores: Any, ids: Any, k: int) -> tuple[list[int], dict[str, Any]]:
    if not 0 <= k <= len(ix):
        raise ValueError(f"Invalid K={k}, n={len(ix)}")
    ordered = sorted(ix, key=lambda i: (-float(scores[i]), str(ids[i])))
    chosen = ordered[:k]
    excluded = ordered[k:]
    boundary = float(scores[chosen[-1]]) if k else None
    ties = sum(float(scores[i]) == boundary for i in ix) if k else 0
    chosen_ties = sum(float(scores[i]) == boundary for i in chosen) if k else 0
    return chosen, {
        "n_candidates": len(ix), "k": k, "boundary_score": boundary,
        "n_equal_boundary_score": ties, "selected_equal_boundary_score": chosen_ties,
        "excluded_equal_boundary_score": ties - chosen_ties,
        "selected_ids_sha256": set_hash([ids[i] for i in chosen]),
        "excluded_ids_sha256": set_hash([ids[i] for i in excluded]),
    }


def independent_metric(y: Any, pred: Any) -> dict[str, Any]:
    import numpy as np

    tn = int(np.count_nonzero((y == 0) & (~pred)))
    fp = int(np.count_nonzero((y == 0) & pred))
    fn = int(np.count_nonzero((y == 1) & (~pred)))
    tp = int(np.count_nonzero((y == 1) & pred))
    safe = lambda a, b: float(a / b) if b else 0.0
    f1p = safe(2 * tp, 2 * tp + fp + fn)
    f1n = safe(2 * tn, 2 * tn + fp + fn)
    return {
        "n": int(len(y)), "actual_positive": int(np.count_nonzero(y == 1)),
        "actual_negative": int(np.count_nonzero(y == 0)),
        "predicted_positive": int(np.count_nonzero(pred)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "macro_f1": (f1p + f1n) / 2.0, "positive_f1": f1p, "negative_f1": f1n,
        "positive_precision": safe(tp, tp + fp), "positive_recall": safe(tp, tp + fn),
        "negative_precision": safe(tn, tn + fn), "negative_recall": safe(tn, tn + fp),
    }


def independently_rebuild(view: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    ids, cycles, origins = view["ids"], view["cycles"], view["origins"]
    missing, y, pa, pp = view["missing"], view["y"], view["pA"], view["pP"]
    present = ~missing
    mix = 0.5 * pp + 0.5 * pa
    preds = {
        "complex_threshold": np.where(missing, pa >= MISSING_THRESHOLD, mix >= PRESENT_THRESHOLD),
        "simple_threshold": np.where(missing, pa >= MISSING_THRESHOLD, pa >= PRESENT_THRESHOLD),
        "simple_topk_cycle_at_complex_count": np.zeros(len(ids), dtype=bool),
        "complex_topk_cycle_at_simple_count": np.zeros(len(ids), dtype=bool),
        "simple_topk_origin_at_complex_count": np.zeros(len(ids), dtype=bool),
        "complex_topk_origin_at_simple_count": np.zeros(len(ids), dtype=bool),
    }
    cycle_details, origin_details = [], []

    def apply(name: str, groups: list[tuple[str, list[int], str]]) -> list[dict[str, Any]]:
        details = []
        for label, ix, key in groups:
            cand = [i for i in ix if present[i]]
            kc = sum(mix[i] >= PRESENT_THRESHOLD for i in cand)
            ks = sum(pa[i] >= PRESENT_THRESHOLD for i in cand)
            a, tie_a = independent_topk(cand, pa, ids, kc)
            b, tie_b = independent_topk(cand, mix, ids, ks)
            n1 = "simple_topk_cycle_at_complex_count" if name == "cycle" else "simple_topk_origin_at_complex_count"
            n2 = "complex_topk_cycle_at_simple_count" if name == "cycle" else "complex_topk_origin_at_simple_count"
            preds[n1][a] = True
            preds[n2][b] = True
            cset = {str(ids[i]) for i in cand if mix[i] >= PRESENT_THRESHOLD}
            sset = {str(ids[i]) for i in cand if pa[i] >= PRESENT_THRESHOLD}
            details.append({
                "group": label, "key": key, "n_rows": len(ix), "n_present": len(cand),
                "k_complex": kc, "k_simple": ks,
                "simple_topk_at_complex_count": tie_a,
                "complex_topk_at_simple_count": tie_b,
                "simple_topk_overlap_with_complex": len({str(ids[i]) for i in a} & cset),
                "complex_topk_overlap_with_simple": len({str(ids[i]) for i in b} & sset),
            })
        return details

    cycles_groups = []
    for origin in sorted(set(int(x) for x in origins.tolist())):
        for cycle in sorted(set(int(x) for x in cycles[origins == origin].tolist())):
            ix = [i for i in range(len(ids)) if int(origins[i]) == origin and int(cycles[i]) == cycle]
            cycles_groups.append((f"E{origin}_cycle_{cycle}", ix, [origin, cycle]))
    cycle_details = apply("cycle", cycles_groups)
    origin_groups = []
    for origin in sorted(set(int(x) for x in origins.tolist())):
        ix = [i for i in range(len(ids)) if int(origins[i]) == origin]
        origin_groups.append((f"E{origin}_origin_block", ix, [origin]))
    origin_details = apply("origin", origin_groups)

    metrics_full = {n: independent_metric(y, preds[n]) for n in POLICIES}
    metrics_present = {n: independent_metric(y[present], preds[n][present]) for n in POLICIES}
    psets = {n: {str(x) for x in ids[present & preds[n]]} for n in POLICIES}
    complex_set, simple_set = psets["complex_threshold"], psets["simple_threshold"]
    overlaps = {}
    for name, values in psets.items():
        uc, us = values | complex_set, values | simple_set
        overlaps[name] = {
            "present_positive_ids_sha256": set_hash(values),
            "overlap_with_complex": len(values & complex_set),
            "jaccard_with_complex": len(values & complex_set) / len(uc) if uc else 1.0,
            "overlap_with_simple": len(values & simple_set),
            "jaccard_with_simple": len(values & simple_set) / len(us) if us else 1.0,
            "changed_rows_vs_complex": len(values ^ complex_set),
            "changed_rows_vs_simple": len(values ^ simple_set),
        }
    from sklearn.metrics import average_precision_score, roc_auc_score
    rank = {
        "present_rows": int(present.sum()),
        "actual_positive": int(np.sum(y[present] == 1)),
        "actual_negative": int(np.sum(y[present] == 0)),
        "simple_pA": {"roc_auc": float(roc_auc_score(y[present], pa[present])),
                      "average_precision": float(average_precision_score(y[present], pa[present]))},
        "complex_half_blend": {"roc_auc": float(roc_auc_score(y[present], mix[present])),
                               "average_precision": float(average_precision_score(y[present], mix[present]))},
    }
    counts = {n: {"present_positive_count": int(np.sum(present & preds[n])),
                  "full_positive_count": int(np.sum(preds[n]))} for n in POLICIES}
    return {
        "operational_full": metrics_full, "present_only": metrics_present,
        "ranking_present_only": rank, "selection_counts": counts,
        "overlap_present_only": overlaps,
        "matching": {"per_cycle": cycle_details, "origin_level_sensitivity": origin_details},
        "ordered_ids_sha256": ordered_hash(ids), "unique_ids": len(set(ids.tolist())),
        "rows": len(ids), "present": int(present.sum()), "missing": int(missing.sum()),
    }


def assert_nested(actual: Any, expected: Any, path: str, checks: list[str], tol: float = 1e-12) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected).issubset(actual):
            raise AssertionError(f"Missing keys at {path}")
        for key, value in expected.items():
            assert_nested(actual[key], value, f"{path}.{key}", checks, tol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"List length mismatch at {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_nested(a, e, f"{path}[{i}]", checks, tol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(expected, float):
            if abs(float(actual) - expected) > tol:
                raise AssertionError(f"Numeric mismatch at {path}: {actual} vs {expected}")
        elif int(actual) != expected:
            raise AssertionError(f"Integer mismatch at {path}: {actual} vs {expected}")
    else:
        if actual != expected:
            raise AssertionError(f"Value mismatch at {path}: {actual!r} vs {expected!r}")
    checks.append(path)


def main() -> None:
    if QA.exists():
        raise FileExistsError(f"Refusing to overwrite independent QA: {QA}")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    if results.get("pre_access_freeze_sha256") != sha_file(FREEZE):
        raise ValueError("Results do not point to the exact committed pre-access freeze")
    verify_lineage(freeze)
    data = load_vectors(freeze)
    checks: list[str] = []
    overlap_h12_e75 = None
    rebuilt = {}
    for name, segments in EXPECTED_VIEWS.items():
        view = make_view(data, segments)
        expected_rows = EXPECTED_ROWS[name]
        if len(view["ids"]) != expected_rows:
            raise AssertionError(f"{name} row count mismatch {len(view['ids'])} != {expected_rows}")
        if len(set(view["ids"].tolist())) != expected_rows:
            raise AssertionError(f"{name} IDs not unique")
        checks.extend([f"{name}.rows={expected_rows}", f"{name}.unique_ids={expected_rows}"])
        rebuilt[name] = independently_rebuild(view)
    ids_h12 = set(make_view(data, EXPECTED_VIEWS["E67_H12"])["ids"].tolist())
    ids_e75 = set(make_view(data, EXPECTED_VIEWS["E75_W6"])["ids"].tolist())
    overlap_h12_e75 = len(ids_h12 & ids_e75)
    if overlap_h12_e75 != EXPECTED_ROWS["E75_W6"]:
        raise AssertionError("E67 H12/E75 W6 expected shared late rows are not fully overlapping")
    checks.append(f"E67_H12_vs_E75_W6_overlap={overlap_h12_e75}")

    for name in EXPECTED_VIEWS:
        stored = results["views"][name]
        expected = rebuilt[name]
        # Compare every numeric and identity field generated independently.
        assert_nested(stored["operational_full"], expected["operational_full"], f"{name}.operational_full", checks)
        assert_nested(stored["present_only"], expected["present_only"], f"{name}.present_only", checks)
        assert_nested(stored["ranking_present_only"], expected["ranking_present_only"], f"{name}.ranking", checks)
        assert_nested(stored["selection_counts"], expected["selection_counts"], f"{name}.selection_counts", checks)
        assert_nested(stored["overlap_present_only"], expected["overlap_present_only"], f"{name}.overlap", checks)
        assert_nested(stored["matching"], expected["matching"], f"{name}.matching", checks)
        assert_nested(stored["denominator"]["ordered_ids_sha256"], expected["ordered_ids_sha256"],
                      f"{name}.ordered_ids_sha256", checks)
        assert_nested(stored["denominator"]["rows"], expected["rows"], f"{name}.rows", checks)
        assert_nested(stored["denominator"]["present"], expected["present"], f"{name}.present", checks)
        assert_nested(stored["denominator"]["missing"], expected["missing"], f"{name}.missing", checks)
        assert_nested(stored["denominator"]["unique_ids"], expected["unique_ids"], f"{name}.unique_ids", checks)
        # Matched policies must preserve counts at every declared matching block.
        for group in expected["matching"]["per_cycle"]:
            if group["simple_topk_at_complex_count"]["k"] != group["k_complex"]:
                raise AssertionError(f"{name}: A02 matched K differs from complex count")
            if group["complex_topk_at_simple_count"]["k"] != group["k_simple"]:
                raise AssertionError(f"{name}: complex matched K differs from simple count")
        checks.append(f"{name}.per_cycle_matched_counts_exact")

    qa = {
        "schema_version": 1,
        "status": "PASS",
        "diagnostic_id": "CLS_SIMPLE_RANK_ALLOCATION_DIAGNOSTIC_V1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier": "standalone; no producer import",
        "pre_access_freeze_sha256": sha_file(FREEZE),
        "results_sha256": sha_file(RESULTS),
        "source_commit": freeze["source_commit"],
        "root_audit": freeze["root_audit"],
        "checks_passed": len(checks),
        "check_names": checks,
        "cross_origin_overlap": {"E67_H12_vs_E75_W6": overlap_h12_e75},
        "unique_pool": {"rows": EXPECTED_ROWS["UNIQUE"], "unique_ids": True,
                        "segments": [[67, 68, 75], [75, 76, 79]]},
        "independent_confusion_count_replay": True,
        "independent_stable_topk_replay": True,
        "ID_order_tie_and_overlap_checks": True,
        "raw_train_label_access": False,
        "test_or_sample_access": False,
        "e79_access": False,
        "new_fits": 0,
        "new_model_inference": 0,
        "kaggle_access": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with QA.open("x", encoding="utf-8") as f:
        json.dump(qa, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"status": qa["status"], "checks_passed": qa["checks_passed"],
                      "results_sha256": qa["results_sha256"], "qa": str(QA)}, indent=2))


if __name__ == "__main__":
    main()
