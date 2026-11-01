"""Zero-fit, all-view subgroup gate audit for cached N01 and N02 pilots.

This amendment reads only official Classification Train and verified pilot
bundles. It never calls an estimator fit method on official rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import optimization_v3_research_next_n01 as n01
from scripts import optimization_v3_research_next_n02 as n02
from src.models.round1_models import _number_array


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/optimization_v3/research_next_v1/results/N01_N02_GUARD_AUDIT/amendment_01"
MIN_GROUP_N = 200
MIN_EACH_CLASS_N = 30
MAX_F1_HARM = 0.01
MAX_POSITIVE_RECALL_HARM = 0.02
MIN_PRIMARY_GAIN = 0.0005
MAX_H12_HARM = 0.001
MAX_WINDOW_HARM = 0.005
MIN_MATCHED_GAIN = 0.0002
ORIGINS = (55, 67)
VIEW_ORDER = ("primary", "W1", "W4", "H12_55", "H12_67", "H12_pooled")
VARIANT_ORDER = ("N01_class_by_volume", "N01_class_only", "N02_sketch", "N02_no_sketch")
EXPECTED_VIEW_ROWS = {
    "primary": 9064, "W1": 4114, "W4": 4950,
    "H12_55": 17685, "H12_67": 20903, "H12_pooled": 38588,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to overwrite different audit evidence: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _event_snapshot(path: Path) -> tuple[str, int, int]:
    raw = path.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    starts = sum(item.get("event") == "FIT_STARTED" for item in events)
    return hashlib.sha256(raw).hexdigest(), len(events), starts


def _merge_cached_origins(
    context: Mapping[str, Any], module: Any, fit_names: Sequence[str],
) -> dict[int, dict[str, np.ndarray]]:
    outputs: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in ORIGINS:
        bundles: dict[str, Mapping[str, np.ndarray]] = {}
        for fit_name in fit_names:
            loaded = module._load_existing_fit(context, cutoff, fit_name)
            if loaded is None:
                raise RuntimeError(f"Missing verified cached pilot bundle: {module.CARD_ID}/{cutoff}/{fit_name}")
            bundles[fit_name] = loaded[0]
        if module.CARD_ID == "N01":
            outputs[cutoff] = module._merge_origin(
                context, cutoff,
                {"class_by_volume": bundles["class_by_volume"], "class_only": bundles["class_only"]},
            )
        else:
            outputs[cutoff] = module._merge_origin(
                context, cutoff,
                {"sketch": bundles["sketch"], "no_sketch": bundles["no_sketch"]},
            )
    return outputs


def _collect(outputs: Mapping[int, Mapping[str, np.ndarray]],
             blocks: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for cutoff, cycle_lo, cycle_hi in blocks:
        record = outputs[int(cutoff)]
        mask = ((record["cycles"] >= int(cycle_lo)) & (record["cycles"] <= int(cycle_hi)))
        parts.append({key: np.asarray(value)[mask] for key, value in record.items()})
    if not parts:
        raise ValueError("An evaluation view must contain at least one origin block")
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(merged["rows"], kind="mergesort")
    merged = {key: value[order] for key, value in merged.items()}
    if len(np.unique(merged["rows"])) != len(merged["rows"]):
        raise RuntimeError("Evaluation view has overlapping source rows")
    return merged


def _metric(y: Any, probability: Any) -> dict[str, float | int]:
    truth = np.asarray(y, dtype=np.int64)
    predicted = (np.asarray(probability, dtype=np.float64) >= 0.50).astype(np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or not np.isin(truth, (0, 1)).all():
        raise ValueError("Independent metric requires aligned binary labels and probabilities")
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    f1_neg = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    return {"n": int(len(truth)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "macro_f1": float(0.5 * (f1_neg + f1_pos)),
            "positive_recall": float(recall_pos)}


def _fit_defined_text_bins(frame: Any, cutoff: int) -> tuple[np.ndarray, list[float]]:
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    finite_fit = parsed[(cycles <= cutoff) & np.isfinite(parsed)]
    bins = np.full(len(frame), -1, dtype=np.int64)
    if not len(finite_fit):
        return bins, []
    edges = np.unique(np.quantile(finite_fit, [0.25, 0.50, 0.75]))
    valid = np.isfinite(parsed)
    bins[valid] = np.searchsorted(edges, parsed[valid], side="right").astype(np.int64)
    return bins, edges.astype(float).tolist()


def _row_hash(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def _view_definitions(views: Mapping[str, Any]) -> dict[str, list[list[int]]]:
    primary = [list(map(int, item)) for item in views["primary"]]
    h12 = [list(map(int, item)) for item in views["h12"]]
    if (len(primary) != 2 or len(h12) != 2 or primary[0][0] != 55 or primary[1][0] != 67
            or h12[0][0] != 55 or h12[1][0] != 67):
        raise RuntimeError("Frozen classification pilot view definitions changed")
    return {
        "primary": primary,
        "W1": [primary[0]],
        "W4": [primary[1]],
        "H12_55": [h12[0]],
        "H12_67": [h12[1]],
        "H12_pooled": h12,
    }


def _candidate_arrays(n01_outputs: Mapping[int, Mapping[str, np.ndarray]],
                      n02_outputs: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Mapping[int, Mapping[str, np.ndarray]]]:
    result: dict[str, Mapping[int, Mapping[str, np.ndarray]]] = {}
    for name in VARIANT_ORDER:
        card = "N01" if name.startswith("N01") else "N02"
        outputs = n01_outputs if card == "N01" else n02_outputs
        key = {"N01_class_by_volume": "candidate", "N01_class_only": "matched",
               "N02_sketch": "candidate", "N02_no_sketch": "matched"}[name]
        result[name] = {cutoff: {**outputs[cutoff], "prediction": outputs[cutoff][key]}
                        for cutoff in ORIGINS}
    return result


def _analyze_view(frame: Any, target: np.ndarray, blocks: Sequence[Sequence[int]],
                  outputs: Mapping[int, Mapping[str, np.ndarray]],
                  variants: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
                  bins_by_origin: Mapping[int, np.ndarray],
                  edges_by_origin: Mapping[int, list[float]],
                  view_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    view = _collect(outputs, blocks)
    variant_views = {name: _collect(variants[name], blocks) for name in VARIANT_ORDER}
    rows = view["rows"].astype(np.int64)
    if len(rows) != EXPECTED_VIEW_ROWS[view_name]:
        raise RuntimeError(f"{view_name} denominator changed: expected {EXPECTED_VIEW_ROWS[view_name]}, got {len(rows)}")
    for name, candidate_view in variant_views.items():
        if not np.array_equal(rows, candidate_view["rows"]):
            raise RuntimeError(f"{view_name}/{name} source rows do not match the frozen comparison population")
    y = target[rows]
    actual_cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
    actual_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
    if not np.array_equal(actual_cycles, view["cycles"].astype(np.int64)):
        raise RuntimeError(f"{view_name} source row/cycle provenance failed")
    if not np.array_equal(actual_ids, view["record_ids"].astype(str)):
        raise RuntimeError(f"{view_name} source row/record_id provenance failed")
    provenance = {
        "n_unique_source_rows": int(len(rows)),
        "source_rows_sha256": _row_hash(rows),
        "record_ids_sha256": hashlib.sha256("\n".join(actual_ids).encode("utf-8")).hexdigest(),
        "cycle_counts": {str(int(cycle)): int(np.sum(actual_cycles == cycle))
                         for cycle in np.unique(actual_cycles)},
        "source_rows_unique": True,
    }
    current_metric = _metric(y, view["current"])
    variant_metrics: dict[str, Any] = {}
    for name in VARIANT_ORDER:
        candidate = variant_views[name]["prediction"]
        metric = _metric(y, candidate)
        variant_metrics[name] = {
            "metric": metric,
            "gain_macro_f1_vs_sparse25": float(metric["macro_f1"] - current_metric["macro_f1"]),
            "positive_recall_gain_vs_sparse25": float(metric["positive_recall"] - current_metric["positive_recall"]),
        }
    # The main candidate/control pair shares exact source rows by construction.
    if view_name == "primary":
        pairs = (("N01_class_by_volume", "N01_class_only"), ("N02_sketch", "N02_no_sketch"))
        for candidate_name, control_name in pairs:
            candidate_metric = variant_metrics[candidate_name]["metric"]["macro_f1"]
            control_metric = variant_metrics[control_name]["metric"]["macro_f1"]
            variant_metrics[candidate_name]["gain_macro_f1_vs_matched_control"] = float(candidate_metric - control_metric)

    group_rows: list[dict[str, Any]] = []
    local = frame.iloc[rows]
    stream_missing = view["missing"].astype(bool)
    origin_values = view["origin"].astype(np.int64)
    labels_by_field: dict[str, np.ndarray] = {
        "entry_friction": local["entry_friction"].astype(str).to_numpy(),
        "stream_availability": np.where(stream_missing, "missing", "present"),
        "fit_defined_text_load_quartile": np.asarray([
            str(int(bins_by_origin[int(origin)][int(row)])) for row, origin in zip(rows, origin_values)
        ], dtype=str),
    }
    for field, labels in labels_by_field.items():
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            group_rows_idx = rows[mask]
            group_y = target[group_rows_idx]
            counts = {"class_0": int(np.sum(group_y == 0)), "class_1": int(np.sum(group_y == 1))}
            n = int(mask.sum())
            supported = n >= MIN_GROUP_N and min(counts.values()) >= MIN_EACH_CLASS_N
            base = _metric(group_y, view["current"][mask])
            for name in VARIANT_ORDER:
                pred = variant_views[name]["prediction"][mask]
                metric = _metric(group_y, pred)
                f1_harm = float(base["macro_f1"] - metric["macro_f1"])
                recall_harm = float(base["positive_recall"] - metric["positive_recall"])
                group_rows.append({
                    "view": view_name, "variant": name, "field": field, "group": str(label),
                    "n": n, "class_0": counts["class_0"], "class_1": counts["class_1"],
                    "supported": bool(supported), "candidate_macro_f1": metric["macro_f1"],
                    "sparse25_macro_f1": base["macro_f1"], "f1_harm_vs_sparse25": f1_harm,
                    "candidate_positive_recall": metric["positive_recall"],
                    "sparse25_positive_recall": base["positive_recall"],
                    "positive_recall_harm_vs_sparse25": recall_harm,
                    "gate_failure_if_candidate": bool(supported and (
                        f1_harm > MAX_F1_HARM or recall_harm > MAX_POSITIVE_RECALL_HARM)),
                    "source_rows_sha256": _row_hash(group_rows_idx),
                    "fit_text_edges_by_origin": (edges_by_origin if field == "fit_defined_text_load_quartile" else None),
                })
    return {"provenance": provenance, "sparse25": current_metric, "variants": variant_metrics}, group_rows


def _gate_result(variant: str, view_results: Mapping[str, Any], group_rows: Sequence[Mapping[str, Any]],
                 *, matched_control: str | None) -> dict[str, Any]:
    primary = view_results["primary"]["variants"][variant]
    h12 = view_results["H12_pooled"]["variants"][variant]
    w1 = view_results["W1"]["variants"][variant]
    w4 = view_results["W4"]["variants"][variant]
    checks: dict[str, bool] = {
        "primary_gain_at_least_0_0005": primary["gain_macro_f1_vs_sparse25"] >= MIN_PRIMARY_GAIN,
        "H12_harm_at_most_0_001": h12["gain_macro_f1_vs_sparse25"] >= -MAX_H12_HARM,
        "W1_harm_at_most_0_005": w1["gain_macro_f1_vs_sparse25"] >= -MAX_WINDOW_HARM,
        "W4_harm_at_most_0_005": w4["gain_macro_f1_vs_sparse25"] >= -MAX_WINDOW_HARM,
    }
    if matched_control is not None:
        matched_gain = primary["metric"]["macro_f1"] - view_results["primary"]["variants"][matched_control]["metric"]["macro_f1"]
        checks["primary_gain_over_matched_at_least_0_0002"] = matched_gain >= MIN_MATCHED_GAIN
    else:
        matched_gain = None
    relevant = [row for row in group_rows if row["variant"] == variant and row["view"] in VIEW_ORDER]
    failures = [dict(row) for row in relevant if row["supported"] and row["gate_failure_if_candidate"]]
    checks["supported_subgroup_guards_all_views"] = not failures
    return {
        "variant": variant,
        "decision": "PASS_DIAGNOSTIC_GATE" if all(checks.values()) else "FAIL_DIAGNOSTIC_GATE",
        "eligible_for_authorized_continuation": bool(all(checks.values())),
        "checks": checks,
        "primary_gain_vs_matched": matched_gain,
        "supported_group_count": int(sum(bool(row["supported"]) for row in relevant)),
        "unsupported_group_count": int(sum(not bool(row["supported"]) for row in relevant)),
        "supported_group_failures": failures,
        "thresholds": {
            "minimum_group_n": MIN_GROUP_N,
            "minimum_each_class_n": MIN_EACH_CLASS_N,
            "maximum_macro_f1_harm": MAX_F1_HARM,
            "maximum_positive_recall_harm": MAX_POSITIVE_RECALL_HARM,
            "minimum_primary_gain": MIN_PRIMARY_GAIN,
            "maximum_H12_harm": MAX_H12_HARM,
            "maximum_primary_window_harm": MAX_WINDOW_HARM,
            "minimum_primary_gain_over_matched": MIN_MATCHED_GAIN if matched_control else None,
        },
    }


def guard_gate_passes(view_gains: Mapping[str, float],
                      supported_group_harms: Sequence[tuple[int, int, int, float, float]],
                      *, matched_gain: float | None = None) -> bool:
    """Small pure gate used by synthetic regression test and production audit."""
    required = {"primary", "W1", "W4", "H12_pooled"}
    if not required.issubset(view_gains):
        return False
    checks = (
        view_gains["primary"] >= MIN_PRIMARY_GAIN,
        view_gains["H12_pooled"] >= -MAX_H12_HARM,
        view_gains["W1"] >= -MAX_WINDOW_HARM,
        view_gains["W4"] >= -MAX_WINDOW_HARM,
        matched_gain is None or matched_gain >= MIN_MATCHED_GAIN,
        all(n < MIN_GROUP_N or min(c0, c1) < MIN_EACH_CLASS_N
            or (f1_harm <= MAX_F1_HARM and recall_harm <= MAX_POSITIVE_RECALL_HARM)
            for n, c0, c1, f1_harm, recall_harm in supported_group_harms),
    )
    return all(checks)


def run_audit() -> dict[str, Any]:
    ledger_path = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
    ledger_before = _event_snapshot(ledger_path)
    n01_frozen, n01_context = n01._verify_freeze()
    n02_frozen, n02_context = n02._verify_freeze()
    if not np.array_equal(n01_context["target"], n02_context["target"]):
        raise RuntimeError("N01 and N02 contexts read different official Train targets")
    if not np.array_equal(n01_context["cycles"], n02_context["cycles"]):
        raise RuntimeError("N01 and N02 contexts read different source cycle arrays")
    if not np.array_equal(n01_context["frame"]["record_id"].astype(str).to_numpy(),
                          n02_context["frame"]["record_id"].astype(str).to_numpy()):
        raise RuntimeError("N01 and N02 contexts read different Train row identities")

    n01_outputs = _merge_cached_origins(n01_context, n01, ("class_by_volume", "class_only"))
    n02_outputs = _merge_cached_origins(n02_context, n02, ("sketch", "no_sketch"))
    for cutoff in ORIGINS:
        for key in ("rows", "record_ids", "cycles", "origin", "current", "missing"):
            if not np.array_equal(n01_outputs[cutoff][key], n02_outputs[cutoff][key]):
                raise RuntimeError(f"N01/N02 shared source/control alignment differs at {cutoff}/{key}")
    variants = _candidate_arrays(n01_outputs, n02_outputs)
    frame = n01_context["frame"]
    target = n01_context["target"].astype(np.int64)
    queue = n01_context["queue"]
    classification_views = queue["views"]["STANDARD"]
    blocks = _view_definitions(classification_views)
    bins_by_origin: dict[int, np.ndarray] = {}
    edges_by_origin: dict[int, list[float]] = {}
    for cutoff in ORIGINS:
        bins_by_origin[cutoff], edges_by_origin[cutoff] = _fit_defined_text_bins(frame, cutoff)
        expected_bins, expected_edges = n01._text_quartiles(frame, cutoff)
        if not np.array_equal(bins_by_origin[cutoff], expected_bins) or edges_by_origin[cutoff] != expected_edges:
            raise RuntimeError(f"Fit-only text_load quartiles differ from frozen N01 rule at {cutoff}")

    view_results: dict[str, Any] = {}
    all_group_rows: list[dict[str, Any]] = []
    for view_name in VIEW_ORDER:
        result, group_rows = _analyze_view(
            frame, target, blocks[view_name], n01_outputs, variants, bins_by_origin,
            edges_by_origin, view_name,
        )
        view_results[view_name] = result
        all_group_rows.extend(group_rows)

    gates = {
        "N01_class_by_volume_original_candidate": _gate_result(
            "N01_class_by_volume", view_results, all_group_rows, matched_control="N01_class_only"),
        "N01_class_only_diagnostic_candidate": _gate_result(
            "N01_class_only", view_results, all_group_rows, matched_control=None),
        "N02_sketch_candidate": _gate_result(
            "N02_sketch", view_results, all_group_rows, matched_control="N02_no_sketch"),
    }
    # The original N01 pilot verdict remains immutable. Its matched-control
    # criterion was defined candidate minus class-only >= 0.0002.
    old_n01_matched_gain = view_results["primary"]["variants"]["N01_class_by_volume"].get(
        "gain_macro_f1_vs_matched_control")
    if old_n01_matched_gain is None:
        old_n01_matched_gain = (
            view_results["primary"]["variants"]["N01_class_by_volume"]["metric"]["macro_f1"]
            - view_results["primary"]["variants"]["N01_class_only"]["metric"]["macro_f1"]
        )
    gates["N01_class_by_volume_original_candidate"]["original_frozen_pilot_matched_control_pass"] = bool(
        old_n01_matched_gain >= MIN_MATCHED_GAIN)
    gates["N01_class_by_volume_original_candidate"]["original_N01_verdict_changed"] = False

    # Hash every cached prediction input and the frozen inputs/source used to
    # reconstruct the metrics. The ledger must be unchanged by this audit.
    source_paths = [
        ROOT / "experiments/optimization_v3/research_next_v1/queue.json",
        ROOT / "reports/optimization_v3/research_next_v1/queue_readiness.json",
        n01.FREEZE_PATH,
        n02.FREEZE_PATH,
        ROOT / "reports/optimization_v3/research_next_v1/results/N01/gate.json",
        ROOT / "reports/optimization_v3/research_next_v1/results/N02/amendment_02/gate.json",
        ROOT / "scripts/optimization_v3_research_next_n01.py",
        ROOT / "scripts/optimization_v3_research_next_n02.py",
        ROOT / "src/models/research_next_n01.py",
        ROOT / "src/models/research_next_n02.py",
        Path(__file__).resolve(),
    ]
    bundle_paths: list[Path] = []
    for cutoff in ORIGINS:
        for fit_name in ("class_by_volume", "class_only"):
            path = n01._bundle_path(cutoff, fit_name)
            bundle_paths.extend((path / "arrays.npz", path / "receipt.json"))
        for fit_name in ("sketch", "no_sketch"):
            path = n02._bundle_path(cutoff, fit_name)
            bundle_paths.extend((path / "arrays.npz", path / "receipt.json"))
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path)
                     for path in source_paths + bundle_paths}
    train_path = n01_context["train_path"]
    source_hashes[str(train_path.relative_to(ROOT))] = _sha256(train_path)
    ledger_after = _event_snapshot(ledger_path)
    if ledger_before != ledger_after:
        raise RuntimeError("Cached audit modified the Research Next ledger or started an estimator fit")

    row_provenance = {name: view_results[name]["provenance"] for name in VIEW_ORDER}
    audit_identity = {
        "card_id": "N01_N02_GUARD_AUDIT",
        "amendment": "01",
        "source_commit": _git_head(),
        "input_hashes": source_hashes,
        "n01_freeze_signature": n01_frozen["freeze_signature"],
        "n02_freeze_signature": n02_frozen["freeze_signature"],
        "ledger_before_after": {"sha256": ledger_before[0], "event_count": ledger_before[1],
                                "fit_started_count": ledger_before[2]},
        "views": blocks,
        "gates": {"minimum_group_n": MIN_GROUP_N, "minimum_each_class_n": MIN_EACH_CLASS_N,
                  "maximum_f1_harm": MAX_F1_HARM,
                  "maximum_positive_recall_harm": MAX_POSITIVE_RECALL_HARM,
                  "minimum_primary_gain": MIN_PRIMARY_GAIN, "maximum_h12_harm": MAX_H12_HARM,
                  "maximum_primary_window_harm": MAX_WINDOW_HARM,
                  "minimum_matched_control_primary_gain": MIN_MATCHED_GAIN},
    }
    report = {
        "schema_version": 1,
        "status": "PASS_ZERO_FIT_AUDIT",
        "card_id": "N01_N02_GUARD_AUDIT",
        "audit_signature": _sha_json(audit_identity),
        "audit_identity": audit_identity,
        "official_estimator_fits": 0,
        "synthetic_estimator_fits": 0,
        "test_files_opened": 0,
        "sample_submission_opened": 0,
        "kaggle_calls": 0,
        "historical_verdicts_changed": False,
        "source_row_provenance": row_provenance,
        "text_load_fit_edges_by_origin": {str(k): v for k, v in edges_by_origin.items()},
        "view_results": view_results,
        "continuation_gates": gates,
        "recommended_next_actions": {
            "N01_original_by_volume_verdict": "PRESERVE_HISTORICAL_RETIRE",
            "N01_class_only": "RUN_FOUR_ORIGIN_DIAGNOSTIC_ONLY" if gates["N01_class_only_diagnostic_candidate"]["eligible_for_authorized_continuation"] else "DO_NOT_FIT",
            "N02_sketch": "RUN_EIGHT_ORIGIN_AND_MATCHED_FITS" if gates["N02_sketch_candidate"]["eligible_for_authorized_continuation"] else "DO_NOT_FIT",
        },
    }
    return {"report": report, "group_rows": all_group_rows}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io

    columns = ["view", "variant", "field", "group", "n", "class_0", "class_1", "supported",
               "candidate_macro_f1", "sparse25_macro_f1", "f1_harm_vs_sparse25",
               "candidate_positive_recall", "sparse25_positive_recall",
               "positive_recall_harm_vs_sparse25", "gate_failure_if_candidate", "source_rows_sha256"]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def main() -> int:
    result = run_audit()
    report = result["report"]
    _write_once(OUT / "results.json", _json_bytes(report))
    _write_once(OUT / "groups.csv", _csv_bytes(result["group_rows"]))
    print(json.dumps({
        "status": report["status"], "audit_signature": report["audit_signature"],
        "official_estimator_fits": report["official_estimator_fits"],
        "decisions": {key: value["decision"] for key, value in report["continuation_gates"].items()},
        "recommended_next_actions": report["recommended_next_actions"],
        "output": str(OUT),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

