"""Independent verifier and Train-only evaluator for RNBR_TARGET_MEDIAN_V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_target_median_v1 as runner  # noqa: E402
from scripts import optimization_v3_rnbr_official as official  # noqa: E402

RESULTS_PATH = ROOT / "reports/optimization_v3/rnbr_target_median_v1/results.json"
VERIFICATION_PATH = ROOT / "reports/optimization_v3/rnbr_target_median_v1/verification.json"
SUMMARY_PATH = ROOT / "reports/optimization_v3/rnbr_target_median_v1/RESULTS.md"


class VerificationError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in "US":
        body = json.dumps(arr.tolist(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return _sha_bytes(body)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _weighted_median(values: np.ndarray, cosine: np.ndarray, rows: np.ndarray) -> float:
    """Independent implementation of the frozen stable cosine-squared median."""
    value = np.asarray(values, dtype=np.float64)
    sim = np.asarray(cosine, dtype=np.float64)
    source = np.asarray(rows, dtype=np.int64)
    if (value.ndim != 1 or value.shape != sim.shape or value.shape != source.shape or not len(value)
            or not np.isfinite(value).all() or not np.isfinite(sim).all()
            or np.any(sim <= 0) or np.any(sim > 1 + 1e-12) or len(np.unique(source)) != len(source)):
        raise VerificationError("Independent weighted-median input validation failed")
    order = np.lexsort((source, value))
    ordered_v, ordered_w = value[order], np.square(sim[order])
    total = float(ordered_w.sum())
    if not np.isfinite(total) or total <= 0:
        raise VerificationError("Independent weighted-median total weight invalid")
    half, tolerance = total / 2.0, total * 1e-12
    cumulative = 0.0
    for j, current in enumerate(ordered_v):
        if j and current == ordered_v[j - 1]:
            continue
        mass = float(ordered_w[ordered_v == current].sum())
        cumulative += mass
        if cumulative > half + tolerance:
            return float(current)
        if abs(cumulative - half) <= tolerance:
            following = ordered_v[ordered_v > current]
            return float((current + following[0]) / 2.0) if len(following) else float(current)
    return float(ordered_v[-1])


def _shuffle_positions(cycles: np.ndarray, r06: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Recreate the fixed cycle x R06-decile seed-42 permutation independently."""
    cycles = np.asarray(cycles, dtype=np.int64)
    r06 = np.asarray(r06, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    deciles = np.minimum(9, np.floor(np.clip(r06, 0, 100) / 10)).astype(np.int8)
    output = np.arange(len(rows), dtype=np.int64)
    rng = np.random.default_rng(42)
    for cycle in np.unique(cycles):
        for decile in range(10):
            positions = np.flatnonzero((cycles == cycle) & (deciles == decile))
            positions = positions[np.argsort(rows[positions], kind="stable")]
            if len(positions) > 1:
                output[positions] = positions[rng.permutation(len(positions))]
    return output


def _predict(donors: np.ndarray, target_memory: np.ndarray, baseline: np.ndarray,
             missing: np.ndarray, neighbor_rows: np.ndarray, cosine: np.ndarray,
             counts: np.ndarray) -> dict[str, np.ndarray]:
    lookup = {int(row): i for i, row in enumerate(np.asarray(donors, dtype=np.int64))}
    median = np.zeros(len(baseline), dtype=np.float64)
    applied = np.zeros(len(baseline), dtype=bool)
    delta = np.zeros(len(baseline), dtype=np.float64)
    for i, count_value in enumerate(counts):
        k = int(count_value)
        if missing[i] or k < 5:
            continue
        active_rows = neighbor_rows[i, :k].astype(np.int64)
        try:
            donor_positions = np.asarray([lookup[int(row)] for row in active_rows], dtype=np.int64)
        except KeyError as exc:
            raise VerificationError("Neighbor source row is outside donor target memory") from exc
        median[i] = _weighted_median(target_memory[donor_positions], cosine[i, :k], active_rows)
        applied[i] = True
        delta[i] = np.clip(0.25 * (median[i] - baseline[i]), -5.0, 5.0)
    prediction = np.clip(baseline + delta, 0.0, 100.0)
    return {"weighted_median_target": median, "median_applied": applied,
            "delta": delta, "prediction": prediction,
            "neighbor_count_used": np.where(applied, counts, 0).astype(np.int16)}


def _read_receipt_and_arrays(fit_id: str, freeze: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = runner._bundle_dir(fit_id)
    receipt_path, archive_path = directory / "receipt.json", directory / "arrays.npz"
    if not receipt_path.is_file() or not archive_path.is_file():
        raise VerificationError(f"Missing saved memory bundle: {fit_id}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {k: v for k, v in receipt.items() if k != "receipt_payload_sha256"}
    if (receipt.get("receipt_payload_sha256") != _sha_bytes(_canonical(body))
            or receipt.get("npz_sha256") != runner._sha_file(archive_path)
            or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or receipt.get("source_commit") != freeze["source_commit"]
            or receipt.get("input_hashes") != freeze.get("input_hashes")
            or receipt.get("fit_id") != fit_id or receipt.get("pilot_id") != runner.PILOT_ID
            or receipt.get("query_targets_read") is not False
            or receipt.get("test_or_sample_submission_opened") is not False
            or receipt.get("kaggle_accessed") is not False):
        raise VerificationError(f"Receipt/lineage verification failed: {fit_id}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for field in receipt["array_hashes"]:
            if field not in archive.files:
                raise VerificationError(f"Missing array {field} in {fit_id}")
            arr = np.asarray(archive[field]).copy()
            if arr.dtype.kind == "O":
                raise VerificationError("Object arrays are prohibited")
            arrays[field] = arr
    # Compare the exact file-level hash inventory using the canonical hash routine.
    if {k: runner._sha_array(v) for k, v in sorted(arrays.items())} != receipt["array_hashes"]:
        raise VerificationError(f"Array hashes differ: {fit_id}")
    origin = int(receipt["origin"])
    if receipt.get("geometry_hashes") != freeze["geometry_hashes"].get(str(origin)):
        raise VerificationError(f"Geometry hashes differ: {fit_id}")
    return receipt, arrays


def _verify_fit(origin: int, arm: str, origin_input: Mapping[str, np.ndarray],
                freeze: Mapping[str, Any]) -> dict[str, Any]:
    fit_id = f"{runner.PILOT_ID}_E{origin}_{arm.upper()}"
    receipt, arrays = _read_receipt_and_arrays(fit_id, freeze)
    for field in ("donor_rows", "donor_ids", "donor_cycles", "donor_source_cutoffs", "donor_r06_oof",
                  "query_rows", "query_ids", "query_cycles", "query_missing", "query_baseline",
                  "neighbor_rows", "neighbor_ids", "neighbor_cosines", "neighbor_counts"):
        if not np.array_equal(arrays[field], origin_input[field]):
            raise VerificationError(f"Frozen input mismatch {fit_id}/{field}")
    actual = official.read_regression_targets(origin_input["donor_rows"], purpose=f"{fit_id}_independent_verify")
    if (not np.array_equal(actual["ids"].astype(str), origin_input["donor_ids"].astype(str))
            or not np.array_equal(actual["cycles"], origin_input["donor_cycles"])
            or not np.array_equal(arrays["donor_targets_actual"], actual["target"])):
        raise VerificationError(f"Donor target lineage mismatch: {fit_id}")
    actual_values = actual["target"]
    if arm == "observed":
        expected_positions = np.arange(len(actual_values), dtype=np.int64)
        memory = actual_values
        expected_sources = origin_input["donor_rows"]
        movement = 0.0
    else:
        expected_positions = _shuffle_positions(origin_input["donor_cycles"],
                                                origin_input["donor_r06_predictions"],
                                                origin_input["donor_rows"])
        memory = actual_values[expected_positions]
        expected_sources = origin_input["donor_rows"][expected_positions]
        movement = float(np.mean(expected_positions != np.arange(len(expected_positions))))
        if movement < 0.80:
            raise VerificationError(f"Shuffled source movement failed for {fit_id}: {movement:.6f}")
    for field, expected in (("target_source_positions", expected_positions),
                            ("target_source_rows", expected_sources),
                            ("donor_targets_used", memory)):
        if not np.array_equal(arrays[field], expected):
            raise VerificationError(f"Target permutation/memory mismatch: {fit_id}/{field}")
    predicted = _predict(origin_input["donor_rows"], memory,
                         origin_input["query_baseline"], origin_input["query_missing"],
                         origin_input["neighbor_rows"], origin_input["neighbor_cosines"],
                         origin_input["neighbor_counts"])
    for field, expected in predicted.items():
        if not np.array_equal(arrays[field], expected):
            raise VerificationError(f"Independent prediction mismatch: {fit_id}/{field}")
    if (np.any(~np.isfinite(arrays["prediction"]))
            or np.any((arrays["prediction"] < 0) | (arrays["prediction"] > 100))
            or not np.array_equal(arrays["prediction"][arrays["query_missing"]],
                                  arrays["query_baseline"][arrays["query_missing"]])):
        raise VerificationError(f"Prediction bounds/fallback failed: {fit_id}")
    return {"fit_id": fit_id, "origin": origin, "arm": arm, "status": "PASS",
            "receipt_file_sha256": runner._sha_file(runner._bundle_dir(fit_id) / "receipt.json"),
            "npz_sha256": receipt["npz_sha256"], "donor_count": int(len(actual_values)),
            "query_count": int(len(arrays["query_rows"])),
            "supported_query_count": int(np.sum(arrays["median_applied"])),
            "shuffle_source_movement": movement,
            "query_predictions_read_only_after_four_outputs_verified": False}


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=np.float64) - np.asarray(prediction, dtype=np.float64))))


def _score(name: str, y: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    values = {arm: _mae(y, pred) for arm, pred in predictions.items()}
    baseline = values["r06"]
    return {"view": name, "n": int(len(y)), "mae": values,
            "gain_vs_r06": {arm: baseline - values[arm] for arm in values if arm != "r06"},
            "gain_observed_vs_target_shuffle": values["target_shuffle"] - values["observed"],
            "gain_observed_vs_original_rnbr": values["original_rnbr"] - values["observed"]}


def _subgroups(y: np.ndarray, missing: np.ndarray, predictions: Mapping[str, np.ndarray], *, view: str) -> dict[str, Any]:
    groups = {
        "signal_stream_missing": np.asarray(missing, dtype=bool),
        "threat_rank_le_5": np.asarray(y) <= 5,
        "threat_rank_ge_95": np.asarray(y) >= 95,
    }
    result: dict[str, Any] = {}
    for name, mask in groups.items():
        n = int(mask.sum())
        harm = _mae(y[mask], predictions["observed"][mask]) - _mae(y[mask], predictions["r06"][mask]) if n else None
        exact = bool(np.array_equal(predictions["observed"][mask], predictions["r06"][mask])) if n else None
        result[name] = {"view": view, "n": n, "observed_harm_vs_r06": harm,
                        "missing_exact_r06": exact if name == "signal_stream_missing" else None,
                        "gate_min_n_100": n >= 100,
                        "gate_harm_le_0_25": bool(harm is not None and harm <= 0.25) if n >= 100 else None,
                        "gate_missing_exact": exact if name == "signal_stream_missing" else None}
    return result


def verify_and_evaluate() -> dict[str, Any]:
    freeze = runner.load_freeze()
    preflight, context = runner._build_targetfree_inputs()
    runner._source_and_input_hashes_match(freeze, preflight)
    events = runner._ledger_events(runner.LEDGER_PATH)
    snapshot = runner.ledger_snapshot(events)
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("reservation_key") == runner.RESERVATION_KEY]
    if (len(reservation) != 1 or reservation[0].get("fits") != 4
            or reservation[0].get("fit_ids") != list(runner.FIT_IDS)
            or snapshot.get("global_cap") != 87 or snapshot.get("fit_starts") != 87
            or snapshot.get("fit_saved") != 84 or snapshot.get("fit_failed") != 3
            or snapshot.get("fit_pending") != 0
            or snapshot.get("active_reservations_by_category") != runner.AMENDED_CAP_SNAPSHOT["effective_category_caps"]):
        raise VerificationError(f"Shared fit ledger did not close at exactly four saved fits: {snapshot}")
    target_fit_events = [e for e in events if e.get("fit_id") in set(runner.FIT_IDS)
                         and e.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED"}]
    if (sum(e["event"] == "FIT_STARTED" for e in target_fit_events) != 4
            or sum(e["event"] == "FIT_SAVED" for e in target_fit_events) != 4
            or any(e["event"] == "FIT_FAILED" for e in target_fit_events)):
        raise VerificationError("Target-median ledger events do not show four starts/four saves/no failures")

    fit_audits = []
    bundle_arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    # Verify every durable output before reading a single query target.
    for origin in runner.ORIGINS:
        for arm in runner.ARMS:
            fit_audits.append(_verify_fit(origin, arm, context["origins"][origin], freeze))
            _, arrays = _read_receipt_and_arrays(f"{runner.PILOT_ID}_E{origin}_{arm.upper()}", freeze)
            bundle_arrays[(origin, arm)] = arrays

    query_rows = np.unique(np.concatenate([context["origins"][o]["query_rows"] for o in runner.ORIGINS])).astype(np.int64)
    query_truth = official.read_regression_targets(query_rows, purpose=f"{runner.PILOT_ID}_evaluation_queries_after_four_verified_fits")
    row_position = {int(row): i for i, row in enumerate(query_rows)}
    per_view: dict[str, dict[str, Any]] = {}
    subgroup_views: dict[str, Any] = {}
    pooled_truth: list[np.ndarray] = []
    pooled_pred: dict[str, list[np.ndarray]] = {key: [] for key in ("r06", "observed", "target_shuffle", "original_rnbr")}
    pooled_missing: list[np.ndarray] = []
    for view, origin, low, high in (("W4", 67, 68, 71), ("H12_E67", 67, 68, 79), ("W6", 75, 76, 79)):
        source = context["origins"][origin]
        select = (source["query_cycles"] >= low) & (source["query_cycles"] <= high)
        rows = source["query_rows"][select]
        qpos = np.asarray([row_position[int(row)] for row in rows], dtype=np.int64)
        y = query_truth["target"][qpos]
        if not np.array_equal(query_truth["ids"][qpos].astype(str), source["query_ids"][select].astype(str)):
            raise VerificationError(f"Query target IDs do not align in {view}")
        observed = bundle_arrays[(origin, "observed")]["prediction"][select]
        shuffled = bundle_arrays[(origin, "shuffled")]["prediction"][select]
        predictions = {"r06": source["query_baseline"][select], "observed": observed,
                       "target_shuffle": shuffled, "original_rnbr": source["original_prediction"][select]}
        per_view[view] = _score(view, y, predictions)
        per_view[view]["coverage_present"] = {
            "present_denominator": int(np.sum(~source["query_missing"][select])),
            "supported_present": int(np.sum(bundle_arrays[(origin, "observed")]["median_applied"][select]
                                             & ~source["query_missing"][select])),
            "fraction": float(np.sum(bundle_arrays[(origin, "observed")]["median_applied"][select]
                                      & ~source["query_missing"][select]) /
                              max(1, int(np.sum(~source["query_missing"][select])))),
        }
        per_view[view]["subgroups"] = _subgroups(y, source["query_missing"][select], predictions, view=view)
        subgroup_views[view] = per_view[view]["subgroups"]
        if view in {"W4", "W6"}:
            pooled_truth.append(y)
            pooled_missing.append(source["query_missing"][select])
            for key in pooled_pred:
                pooled_pred[key].append(predictions[key])
    pooled_y = np.concatenate(pooled_truth)
    pooled_predictions = {key: np.concatenate(parts) for key, parts in pooled_pred.items()}
    pooled = _score("POOLED_W4_W6", pooled_y, pooled_predictions)
    pooled["subgroups"] = _subgroups(pooled_y, np.concatenate(pooled_missing), pooled_predictions, view="POOLED_W4_W6")

    g = pooled["gain_vs_r06"]
    gates = {
        "W4_gain_vs_R06_nonnegative": per_view["W4"]["gain_vs_r06"]["observed"] >= 0,
        "W6_gain_vs_R06_nonnegative": per_view["W6"]["gain_vs_r06"]["observed"] >= 0,
        "pooled_W4_W6_gain_vs_R06_at_least_0_03": g["observed"] >= 0.03,
        "H12_E67_gain_vs_R06_nonnegative": per_view["H12_E67"]["gain_vs_r06"]["observed"] >= 0,
        "pooled_gain_vs_target_shuffle_at_least_0_01": pooled["gain_observed_vs_target_shuffle"] >= 0.01,
        "H12_gain_vs_target_shuffle_nonnegative": per_view["H12_E67"]["gain_observed_vs_target_shuffle"] >= 0,
        "pooled_gain_vs_original_RNBR_at_least_0_03": pooled["gain_observed_vs_original_rnbr"] >= 0.03,
        "H12_gain_vs_original_RNBR_nonnegative": per_view["H12_E67"]["gain_observed_vs_original_rnbr"] >= 0,
    }
    for view_name, subgroup_map in (("POOLED_W4_W6", pooled["subgroups"]), ("H12_E67", subgroup_views["H12_E67"])):
        for group_name, item in subgroup_map.items():
            if item["n"] < 100 or item["observed_harm_vs_r06"] is None:
                gates[f"{view_name}_{group_name}_minimum_n_100"] = False
            gates[f"{view_name}_{group_name}_harm_at_most_0_25"] = item["observed_harm_vs_r06"] is not None and item["observed_harm_vs_r06"] <= 0.25
            if group_name == "signal_stream_missing":
                gates[f"{view_name}_missing_exact_R06"] = item["missing_exact_r06"] is True
    integrity_pass = True  # Reached only after freeze, lineage, receipt, ledger, and independent formula checks pass.
    passed = integrity_pass and all(gates.values())
    status = "ADVANCE_PILOT_ONLY" if passed else "FAIL_RETIRE_FUNCTIONAL"
    verification = {
        "status": "PASS_INDEPENDENT_INTEGRITY_AND_FORMULA" if integrity_pass else "FAIL",
        "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
        "fits": fit_audits, "ledger_snapshot": snapshot,
        "ledger_sha256": runner._sha_file(runner.LEDGER_PATH),
        "query_targets_read_after_four_memory_outputs_verified": True,
        "test_or_sample_submission_opened": False, "kaggle_accessed": False,
        "classification_labels_or_probabilities_read": False,
        "independent_formula_implementation": True,
    }
    results = {
        "status": status, "diagnostic_only": True, "continuation_qualified": False,
        "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
        "estimand": "all-row MAE improvement, MAE comparator minus candidate MAE; positive values favor candidate",
        "primary_pooled_W4_W6": pooled, "views": per_view, "gates": gates,
        "all_gates_pass": all(gates.values()), "integrity_pass": integrity_pass,
        "protected_subgroups": {"pooled_W4_W6": pooled["subgroups"], "H12_E67": subgroup_views["H12_E67"]},
        "interpretation": "A pass is a pilot-only signal; a failure retires this functional. No continuation or submission is authorized.",
        "limitations": ["H12 overlaps W4 and is a fixed-origin diagnostic, not an independent sample.",
                        "All development rows and prior public feedback are adaptive.",
                        "Subgroup membership is evaluation-only; no causal interpretation."],
    }
    runner._write_json_once(VERIFICATION_PATH, verification)
    runner._write_json_once(RESULTS_PATH, results)
    runner._write_once(SUMMARY_PATH, _markdown(results, verification).encode("utf-8"))
    return {"verification": verification, "results": results}


def _markdown(results: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    lines = ["# RNBR Target Median V1 — Results", "",
             f"**Verdict:** `{results['status']}` (diagnostic-only; continuation qualified: no).", "",
             "## Primary pooled W4 + W6", "",
             "| Arm | MAE | Gain vs R06 |", "|---|---:|---:|"]
    primary = results["primary_pooled_W4_W6"]
    for arm in ("r06", "observed", "target_shuffle", "original_rnbr"):
        gain = "—" if arm == "r06" else f"{primary['gain_vs_r06'].get(arm, float('nan')):+.4f}"
        lines.append(f"| {arm} | {primary['mae'][arm]:.4f} | {gain} |")
    lines += ["", f"Observed vs matched target-shuffle gain: {primary['gain_observed_vs_target_shuffle']:+.4f} MAE.",
              f"Observed vs original RNBR gain: {primary['gain_observed_vs_original_rnbr']:+.4f} MAE.", "",
              "## Fixed views", "", "| View | n | R06 MAE | Observed MAE | Shuffle MAE | Original RNBR MAE | Gain vs R06 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in results["views"].items():
        lines.append(f"| {name} | {row['n']} | {row['mae']['r06']:.4f} | {row['mae']['observed']:.4f} | {row['mae']['target_shuffle']:.4f} | {row['mae']['original_rnbr']:.4f} | {row['gain_vs_r06']['observed']:+.4f} |")
    lines += ["", "## Gates", "", "| Gate | Result |", "|---|---|"]
    lines.extend(f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in results["gates"].items())
    lines += ["", "## Protected subgroup guards", ""]
    for view_name, items in results["protected_subgroups"].items():
        lines += [f"### {view_name}", "", "| Group | n | MAE harm vs R06 | Guard |", "|---|---:|---:|---|"]
        for name, row in items.items():
            harm = "n/a" if row["observed_harm_vs_r06"] is None else f"{row['observed_harm_vs_r06']:+.4f}"
            lines.append(f"| {name} | {row['n']} | {harm} | {'PASS' if row['gate_harm_le_0_25'] else 'FAIL'} |")
    lines += ["", "## Audit boundary", "",
              f"Independent verification: `{verification['status']}`; four memory fits: {len(verification['fits'])}; ledger starts/saves/pending: {verification['ledger_snapshot']['fit_starts']}/{verification['ledger_snapshot']['fit_saved']}/{verification['ledger_snapshot']['fit_pending']}.",
              "Query Train targets were read only after all four memory bundles passed independent receipt, lineage, shuffle, and formula checks. Test, SampleSubmission, Kaggle, and Classification labels/probabilities were not accessed.", "",
              "This outcome does not change the original RNBR verdict, and it does not authorize continuation or a Kaggle submission.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    outcome = verify_and_evaluate()
    print(json.dumps({"status": outcome["results"]["status"],
                      "results_path": RESULTS_PATH.as_posix(),
                      "verification_path": VERIFICATION_PATH.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
