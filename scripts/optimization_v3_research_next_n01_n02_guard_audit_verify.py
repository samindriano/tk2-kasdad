"""Independent cache-to-report verification for the N01/N02 subgroup audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import optimization_v3_research_next_n01 as n01
from scripts import optimization_v3_research_next_n02 as n02
from src.models.round1_models import _number_array


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "reports/optimization_v3/research_next_v1/results/N01_N02_GUARD_AUDIT/amendment_01"
MIN_N, MIN_CLASS_N = 200, 30
MAX_F1_HARM, MAX_RECALL_HARM = 0.01, 0.02
ORIGINS = (55, 67)
VIEWS = {
    "primary": [(55, 56, 59), (67, 68, 71)],
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "H12_55": [(55, 56, 67)],
    "H12_67": [(67, 68, 79)],
    "H12_pooled": [(55, 56, 67), (67, 68, 79)],
}
VARIANTS = ("N01_class_by_volume", "N01_class_only", "N02_sketch", "N02_no_sketch")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_hash(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def _event_snapshot(path: Path) -> tuple[str, int, int]:
    raw = path.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return hashlib.sha256(raw).hexdigest(), len(events), sum(x.get("event") == "FIT_STARTED" for x in events)


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.int64)
    predicted = (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int64)
    tn = int(np.count_nonzero((truth == 0) & (predicted == 0)))
    fp = int(np.count_nonzero((truth == 0) & (predicted == 1)))
    fn = int(np.count_nonzero((truth == 1) & (predicted == 0)))
    tp = int(np.count_nonzero((truth == 1) & (predicted == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(truth), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "macro_f1": 0.5 * (f0 + f1), "positive_recall": recall}


def _load_raw_fit_arrays(module: Any, context: Mapping[str, Any], cutoff: int,
                         names: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for name in names:
        item = module._load_existing_fit(context, cutoff, name)
        if item is None:
            raise RuntimeError(f"Missing historical fit bundle {module.CARD_ID}/{cutoff}/{name}")
        result[name] = item[0]
    return result


def _compose_n01(context: Mapping[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    a = _load_raw_fit_arrays(n01, context, cutoff, ("class_by_volume", "class_only"))
    bv, co = a["class_by_volume"], a["class_only"]
    rows = bv["score_rows"].astype(np.int64)
    missing = bv["stream_missing"].astype(bool)
    c2b = bv["control__c2b_probability"].astype(np.float64)
    cb = bv["control__cb_probability"].astype(np.float64)
    sparse = bv["control__sparse_baseline_raw"].astype(np.float64)
    candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * bv["raw_prediction"])
    matched = np.where(missing, cb, 0.75 * c2b + 0.25 * co["raw_prediction"])
    current = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse)
    return {"rows": rows, "record_ids": bv["score_record_ids"].astype(str),
            "cycles": bv["cycle_code"].astype(np.int64), "origin": np.full(len(rows), cutoff),
            "missing": missing, "current": current,
            "N01_class_by_volume": candidate, "N01_class_only": matched}


def _compose_n02(context: Mapping[str, Any], cutoff: int) -> dict[str, np.ndarray]:
    a = _load_raw_fit_arrays(n02, context, cutoff, ("sketch", "no_sketch"))
    sk, ns = a["sketch"], a["no_sketch"]
    rows = sk["score_rows"].astype(np.int64)
    missing = sk["stream_missing"].astype(bool)
    base = sk["control__current_route"].astype(np.float64)
    cb = sk["control__cb_probability"].astype(np.float64)
    sketch = np.where(missing, cb, 0.75 * base + 0.25 * sk["raw_prediction"])
    no_sketch = np.where(missing, cb, 0.75 * base + 0.25 * ns["raw_prediction"])
    current = np.where(missing, cb, base)
    return {"rows": rows, "record_ids": sk["score_record_ids"].astype(str),
            "cycles": sk["cycle_code"].astype(np.int64), "origin": np.full(len(rows), cutoff),
            "missing": missing, "current": current,
            "N02_sketch": sketch, "N02_no_sketch": no_sketch}


def _collect(records: Mapping[int, Mapping[str, np.ndarray]], blocks: Sequence[Sequence[int]],
             field: str) -> np.ndarray:
    chunks = []
    rows = []
    for cutoff, lo, hi in blocks:
        rec = records[int(cutoff)]
        mask = (rec["cycles"] >= lo) & (rec["cycles"] <= hi)
        rows.append(rec["rows"][mask])
        chunks.append(rec[field][mask])
    all_rows = np.concatenate(rows)
    order = np.argsort(all_rows, kind="mergesort")
    if len(np.unique(all_rows)) != len(all_rows):
        raise RuntimeError("Independent verifier found duplicated source rows")
    return np.concatenate(chunks)[order]


def _origin_aligned_records(records: Mapping[int, Mapping[str, np.ndarray]],
                            blocks: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    row_chunks, origin_chunks = [], []
    for cutoff, lo, hi in blocks:
        rec = records[int(cutoff)]
        mask = (rec["cycles"] >= lo) & (rec["cycles"] <= hi)
        row_chunks.append(rec["rows"][mask])
        origin_chunks.append(rec["origin"][mask])
    rows, origins = np.concatenate(row_chunks), np.concatenate(origin_chunks)
    order = np.argsort(rows, kind="mergesort")
    return rows[order], origins[order]


def _fit_bins(frame: Any, cutoff: int) -> tuple[np.ndarray, list[float]]:
    values, _missing, _invalid = _number_array(frame["text_load"])
    fit = frame["cycle_code"].to_numpy(dtype=np.int64) <= cutoff
    finite_fit = values[fit & np.isfinite(values)]
    output = np.full(len(frame), -1, dtype=np.int64)
    if not len(finite_fit):
        return output, []
    edges = np.unique(np.quantile(finite_fit, [0.25, 0.50, 0.75]))
    valid = np.isfinite(values)
    output[valid] = np.searchsorted(edges, values[valid], side="right").astype(np.int64)
    return output, edges.astype(float).tolist()


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-12


def verify() -> dict[str, Any]:
    result_path = AUDIT_DIR / "results.json"
    group_path = AUDIT_DIR / "groups.csv"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    expected_ledger = report["audit_identity"]["ledger_before_after"]["sha256"]
    ledger_path = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
    ledger_before = _event_snapshot(ledger_path)
    if ledger_before[0] != expected_ledger:
        raise RuntimeError("Research Next ledger changed since cached subgroup audit")
    for relative, expected in report["audit_identity"]["input_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"Cached subgroup audit input hash changed: {relative}")

    n01_frozen, c1 = n01._verify_freeze()
    n02_frozen, c2 = n02._verify_freeze()
    if n01_frozen["freeze_signature"] != report["audit_identity"]["n01_freeze_signature"]:
        raise RuntimeError("Independent verifier sees a different frozen N01 pilot")
    if n02_frozen["freeze_signature"] != report["audit_identity"]["n02_freeze_signature"]:
        raise RuntimeError("Independent verifier sees a different frozen N02 pilot")
    if not np.array_equal(c1["target"], c2["target"]):
        raise RuntimeError("N01/N02 official Train targets are not identical")
    y = c1["target"].astype(np.int64)
    frame = c1["frame"]
    records: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in ORIGINS:
        a, b = _compose_n01(c1, cutoff), _compose_n02(c2, cutoff)
        for key in ("rows", "record_ids", "cycles", "missing", "current"):
            if not np.array_equal(a[key], b[key]):
                raise RuntimeError(f"Independent N01/N02 alignment or sparse25 mismatch {cutoff}/{key}")
        records[cutoff] = {**a, "N02_sketch": b["N02_sketch"], "N02_no_sketch": b["N02_no_sketch"]}

    with group_path.open("r", encoding="utf-8", newline="") as handle:
        saved_groups = {(r["view"], r["variant"], r["field"], r["group"]): r for r in csv.DictReader(handle)}
    verification_views: dict[str, Any] = {}
    computed_groups = 0
    for view_name, blocks in VIEWS.items():
        rows, origins = _origin_aligned_records(records, blocks)
        cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
        ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"Verifier found overlapping rows in {view_name}")
        if not np.array_equal(cycles, _collect(records, blocks, "cycles")):
            raise RuntimeError(f"Verifier cycle row mapping failed in {view_name}")
        if hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest() != report["source_row_provenance"][view_name]["source_rows_sha256"]:
            raise RuntimeError(f"Verifier source-row hash changed in {view_name}")
        truth = y[rows]
        current = _collect(records, blocks, "current")
        current_metric = _metric(truth, current)
        variant_views = {}
        for variant in VARIANTS:
            pred = _collect(records, blocks, variant)
            metric = _metric(truth, pred)
            saved = report["view_results"][view_name]["variants"][variant]
            if (metric["tn"] != saved["metric"]["tn"] or metric["fp"] != saved["metric"]["fp"]
                    or metric["fn"] != saved["metric"]["fn"] or metric["tp"] != saved["metric"]["tp"]
                    or not _close(metric["macro_f1"], saved["metric"]["macro_f1"])
                    or not _close(metric["positive_recall"], saved["metric"]["positive_recall"])):
                raise RuntimeError(f"Independent metric mismatch for {view_name}/{variant}")
            variant_views[variant] = pred

        bins = {cutoff: _fit_bins(frame, cutoff) for cutoff in ORIGINS}
        labels_by_field = {
            "entry_friction": frame.iloc[rows]["entry_friction"].astype(str).to_numpy(),
            "stream_availability": np.where(_collect(records, blocks, "missing"), "missing", "present"),
            "fit_defined_text_load_quartile": np.asarray([
                str(int(bins[int(origin)][0][int(row)])) for row, origin in zip(rows, origins)
            ], dtype=str),
        }
        for field, labels in labels_by_field.items():
            for label in sorted(set(labels.tolist())):
                mask = labels == label
                grow = rows[mask]
                gy = y[grow]
                c0, c1n = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                support = len(grow) >= MIN_N and min(c0, c1n) >= MIN_CLASS_N
                base = _metric(gy, current[mask])
                for variant in VARIANTS:
                    key = (view_name, variant, field, str(label))
                    if key not in saved_groups:
                        raise RuntimeError(f"Independent verifier missing group row {key}")
                    row = saved_groups[key]
                    metric = _metric(gy, variant_views[variant][mask])
                    f1_harm = base["macro_f1"] - metric["macro_f1"]
                    recall_harm = base["positive_recall"] - metric["positive_recall"]
                    if (int(row["n"]) != len(grow) or int(row["class_0"]) != c0 or int(row["class_1"]) != c1n
                            or (row["supported"].lower() == "true") != support
                            or not _close(float(row["candidate_macro_f1"]), metric["macro_f1"])
                            or not _close(float(row["sparse25_macro_f1"]), base["macro_f1"])
                            or not _close(float(row["f1_harm_vs_sparse25"]), f1_harm)
                            or not _close(float(row["positive_recall_harm_vs_sparse25"]), recall_harm)
                            or row["source_rows_sha256"] != hashlib.sha256(grow.astype("<i8").tobytes()).hexdigest()):
                        raise RuntimeError(f"Independent subgroup mismatch for {key}")
                    computed_groups += 1
        verification_views[view_name] = {
            "rows": int(len(rows)),
            "source_rows_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
            "sparse25": current_metric,
        }

    n01_classonly_gate = report["continuation_gates"]["N01_class_only_diagnostic_candidate"]
    n02_sketch_gate = report["continuation_gates"]["N02_sketch_candidate"]
    if (n01_classonly_gate["eligible_for_authorized_continuation"] is not True
            or n02_sketch_gate["eligible_for_authorized_continuation"] is not False):
        raise RuntimeError("Corrected continuation gates differ from verified cached subgroup evidence")
    ledger_after = _event_snapshot(ledger_path)
    if ledger_after != ledger_before:
        raise RuntimeError("Independent verifier changed budget ledger")
    return {
        "status": "PASS_INDEPENDENT_ZERO_FIT_VERIFICATION",
        "audit_signature": report["audit_signature"],
        "source_commit": report["audit_identity"]["source_commit"],
        "official_estimator_fits": 0,
        "ledger_unchanged": True,
        "ledger_sha256": ledger_after[0],
        "n01_original_verdict_changed": False,
        "n01_class_only_diagnostic_gate": "PASS",
        "n02_sketch_corrected_gate": "FAIL_H12_TEXT_LOAD_Q4_RECALL_HARM",
        "verified_view_count": len(verification_views),
        "verified_group_rows": computed_groups,
        "views": verification_views,
        "verifier_source_sha256": _sha(Path(__file__).resolve()),
    }


def main() -> int:
    result = verify()
    output = AUDIT_DIR / "verification.json"
    payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if output.exists() and output.read_bytes() != payload:
        raise RuntimeError(f"Refusing to overwrite different verification result: {output}")
    if not output.exists():
        output.write_bytes(payload)
    print(json.dumps({"status": result["status"], "verified_view_count": result["verified_view_count"],
                      "verified_group_rows": result["verified_group_rows"],
                      "N01_class_only": result["n01_class_only_diagnostic_gate"],
                      "N02_sketch": result["n02_sketch_corrected_gate"],
                      "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

