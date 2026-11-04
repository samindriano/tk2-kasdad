"""Zero-fit diagnostics for the frozen direct-position regression pilot.

Reads only already-scored E67/E75 Regression Train query targets and saved
prediction bundles. It never loads a fitted model or calls predict/fit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata


DIRECT_CARD = "REG_POSITION_DIRECT_L1_V1"
RESIDUAL_CARD = "REG_POSITION_L1_V1"
ORIGINS = (67, 75)
ARMS = ("BAG", "REAL", "SHUFFLED")
TRAIN_REL = Path("data/raw/regression/Train.csv")
TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
DIRECT_REPORT_REL = Path("reports/optimization_v3/reg_position_direct_l1_v1")
RESIDUAL_REPORT_REL = Path("reports/optimization_v3/reg_position_l1_v1")
DIRECT_RUN_REL = Path("experiments/runs/reg_position_direct_l1_v1")
RESIDUAL_RUN_REL = Path("experiments/runs/reg_position_l1_v1")
TRANSPORT_REL = Path("reports/optimization_v3/reg_position_transport_diagnostic_v1")
OUTPUT_REL = Path("reports/optimization_v3/reg_position_failure_mechanism_v1")
DIRECT_FREEZE_SIGNATURE = "80614c292413e00fd4234bb1d4da4f3b40f2aea019bb6d510dab035e356b4003"
RESIDUAL_FREEZE_SIGNATURE = "1982463561610456c516d3ac3c913f0be38aca6bd2955e2813960f3e50e03ca5"
EXPECTED_DIRECT_TARGET_SHA256 = "b7c22d8af92d0290ffd96615afedf5e3f6451e8bc204e9297368db33c40c9796"
MIN_GROUP_N = 100
TOL = 1e-10


class AuditError(RuntimeError):
    """A frozen input or arithmetic contract failed closed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(canonical_bytes({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(canonical_bytes(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"Expected a JSON object: {path}")
    return value


def verify_body_signature(value: dict[str, Any], key: str, label: str) -> None:
    if key not in value or value[key] != canonical_sha({k: v for k, v in value.items() if k != key}):
        raise AuditError(f"{label} signature failed")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    require(actual.shape == predicted.shape and actual.size > 0, "MAE inputs are empty or misaligned")
    require(np.isfinite(actual).all() and np.isfinite(predicted).all(), "MAE inputs are nonfinite")
    return float(np.mean(np.abs(actual - predicted)))


def spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.size < 2:
        return float("nan")
    left = rankdata(actual, method="average")
    right = rankdata(predicted, method="average")
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def rank50_map(raw: np.ndarray, cycles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the frozen per-cycle half raw / half percentile map."""
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    require(raw.ndim == cycles.ndim == 1 and raw.shape == cycles.shape,
            "rank50 raw scores and cycles are misaligned")
    require(np.isfinite(raw).all(), "rank50 raw scores are nonfinite")
    percentile = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return 0.5 * raw + 0.5 * percentile, percentile


def direct_policy(baseline: np.ndarray, mapped: np.ndarray, missing: np.ndarray) -> np.ndarray:
    result = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    result = np.asarray(result, dtype=np.float64)
    result[np.asarray(missing, dtype=bool)] = np.asarray(baseline, dtype=np.float64)[missing]
    return result


def residual_policy(baseline: np.ndarray, head: np.ndarray,
                    missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.clip(0.25 * np.asarray(head, dtype=np.float64), -5.0, 5.0)
    prediction = np.clip(np.asarray(baseline, dtype=np.float64) + delta, 0.0, 100.0)
    mask = np.asarray(missing, dtype=bool)
    delta = np.asarray(delta, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    delta[mask] = 0.0
    prediction[mask] = np.asarray(baseline, dtype=np.float64)[mask]
    return delta, prediction


def verify_alignment(rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
                     expected_rows: np.ndarray, expected_ids: np.ndarray,
                     expected_cycles: np.ndarray, label: str) -> None:
    if (not np.array_equal(np.asarray(rows), np.asarray(expected_rows))
            or not np.array_equal(np.asarray(ids, dtype=str), np.asarray(expected_ids, dtype=str))
            or not np.array_equal(np.asarray(cycles), np.asarray(expected_cycles))):
        raise AuditError(f"{label} row / record_id / cycle alignment failed")


def read_scored_query(source_root: Path, selected_rows: np.ndarray,
                      expected_ids: np.ndarray, expected_cycles: np.ndarray,
                      expected_total_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse features and targets only on already-scored cycles 68-79.

    Unselected physical lines are skipped as raw bytes without decoding or
    tokenizing, following the pilot's target reader boundary.
    """
    path = source_root / TRAIN_REL
    require(path.is_file(), f"Regression Train is missing: {path}")
    source_hash = sha_file(path)
    require(source_hash == TRAIN_SHA256, "Regression Train hash differs from frozen source")
    rows = np.asarray(selected_rows, dtype=np.int64)
    ids_expected = np.asarray(expected_ids, dtype=str)
    cycles_expected = np.asarray(expected_cycles, dtype=np.int64)
    require(rows.ndim == 1 and rows.size == ids_expected.size
            and np.array_equal(rows, np.sort(rows)) and len(set(rows.tolist())) == len(rows),
            "Selected query rows are not a unique ordered physical-row vector")
    require(np.all(cycles_expected > 67) and int(cycles_expected.min()) == 68
            and int(cycles_expected.max()) == 79,
            "Selected labels fall outside already-scored cycles 68-79")
    wanted = set(int(x) for x in rows)
    id_column = cycle_column = text_load_column = target_column = None
    parsed_rows: list[int] = []
    parsed_ids: list[str] = []
    parsed_cycles: list[int] = []
    parsed_text_load: list[float] = []
    parsed_targets: list[float] = []
    physical_lines = 0
    with path.open("rb") as stream:
        for physical_line, raw_line in enumerate(stream):
            physical_lines += 1
            if physical_line == 0:
                header = next(csv.reader([raw_line.decode("utf-8-sig")], strict=True))
                require(len(header) == len(set(header)), "Regression Train has duplicate columns")
                for name in ("record_id", "cycle_code", "text_load", "threat_rank"):
                    require(name in header, f"Regression Train lacks {name}")
                id_column, cycle_column = header.index("record_id"), header.index("cycle_code")
                text_load_column, target_column = header.index("text_load"), header.index("threat_rank")
                continue
            row_index = physical_line - 1
            if row_index not in wanted:
                continue
            fields = next(csv.reader([raw_line.decode("utf-8")], strict=True))
            require(len(fields) == len(header), f"Selected source row {row_index} has a bad CSV shape")
            try:
                cycle = int(fields[cycle_column])
                text_load = float(fields[text_load_column])
                target = float(fields[target_column])
            except (TypeError, ValueError) as exc:
                raise AuditError(f"Selected query row {row_index} has invalid feature/target data") from exc
            require(cycle > 67 and 0.0 <= target <= 100.0 and np.isfinite(text_load),
                    f"Selected query row {row_index} violates the frozen query range")
            parsed_rows.append(row_index)
            parsed_ids.append(fields[id_column])
            parsed_cycles.append(cycle)
            parsed_text_load.append(text_load)
            parsed_targets.append(target)
    require(physical_lines == expected_total_rows + 1, "Regression Train physical line count changed")
    require(np.array_equal(np.asarray(parsed_rows, dtype=np.int64), rows),
            "Selected query targets were not parsed exactly once in order")
    verify_alignment(rows, np.asarray(parsed_ids, dtype=str), np.asarray(parsed_cycles, dtype=np.int64),
                     rows, ids_expected, cycles_expected, "Regression Train selected query")
    return (np.asarray(parsed_targets, dtype=np.float64),
            np.asarray(parsed_text_load, dtype=np.float64),
            np.asarray(parsed_rows, dtype=np.int64),
            {"path": TRAIN_REL.as_posix(), "sha256": source_hash,
             "physical_rows": expected_total_rows, "query_target_rows_parsed": len(rows),
             "fit_target_rows_parsed": 0, "selected_cycle_min": int(cycles_expected.min()),
             "selected_cycle_max": int(cycles_expected.max()),
             "query_rows_source_positions_sha256": sha_array(rows),
             "query_record_ids_sha256": sha_array(ids_expected),
             "query_cycles_sha256": sha_array(cycles_expected),
             "query_target_values_sha256": sha_array(np.asarray(parsed_targets, dtype=np.float64))})


def load_direct_bundle(saved_root: Path, report_root: Path, freeze: dict[str, Any],
                       prelabel: dict[str, Any], origin: int, arm: str) -> dict[str, Any]:
    fit_id = f"{DIRECT_CARD}_E{origin}_{arm}"
    path = saved_root / DIRECT_RUN_REL / "predictions" / f"{fit_id}.npz"
    receipt_path = path.with_suffix(".json")
    require(path.is_file() and receipt_path.is_file(), f"Missing saved direct bundle: {fit_id}")
    receipt = read_json(receipt_path)
    verify_body_signature(receipt, "receipt_sha256", f"{fit_id} receipt")
    bundle_hash = sha_file(path)
    receipt_hash = sha_file(receipt_path)
    require(receipt.get("fit_id") == fit_id and receipt.get("freeze_signature") == freeze["freeze_signature"]
            and receipt.get("bundle_sha256") == bundle_hash and receipt.get("query_targets_read") is False,
            f"{fit_id} receipt does not bind the untouched saved bundle")
    pre = prelabel["predictions_verified"][str(origin)][arm]
    require(pre.get("bundle_sha256") == bundle_hash and pre.get("receipt_sha256") == receipt_hash,
            f"{fit_id} does not match the signed prelabel verification")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"origin", "arm", "fit_rows", "query_rows", "query_record_ids", "query_cycles",
                "baseline", "raw_auxiliary", "mapped_auxiliary", "prediction", "missing_mask"}
    require(required.issubset(arrays), f"{fit_id} is missing required arrays")
    require(int(arrays["origin"][0]) == origin and str(arrays["arm"][0]) == arm,
            f"{fit_id} bundle identity fields differ")
    return {"family": "DIRECT", "origin": origin, "arm": arm, "fit_id": fit_id,
            "path": path, "receipt_path": receipt_path, "bundle_sha256": bundle_hash,
            "receipt_file_sha256": receipt_hash, **arrays}


def load_residual_bundle(saved_root: Path, legacy_hashes: dict[str, str],
                         freeze: dict[str, Any], origin: int, arm: str) -> dict[str, Any]:
    fit_id = f"{RESIDUAL_CARD}_E{origin}_{arm}"
    path = saved_root / RESIDUAL_RUN_REL / f"{fit_id}.npz"
    receipt_path = saved_root / RESIDUAL_RUN_REL / f"{fit_id}.receipt.json"
    require(path.is_file() and receipt_path.is_file(), f"Missing saved residual bundle: {fit_id}")
    receipt = read_json(receipt_path)
    verify_body_signature(receipt, "receipt_sha256", f"{fit_id} receipt")
    bundle_hash = sha_file(path)
    receipt_hash = sha_file(receipt_path)
    require(legacy_hashes.get(f"{fit_id}.prediction") == bundle_hash
            and legacy_hashes.get(f"{fit_id}.receipt") == receipt_hash,
            f"{fit_id} bundle or receipt differs from the prior transport audit hash")
    require(receipt.get("fit_id") == fit_id and receipt.get("origin") == origin
            and receipt.get("arm") == arm and receipt.get("freeze_signature") == freeze["freeze_signature"]
            and receipt.get("predictions_sha256") == bundle_hash
            and receipt.get("query_targets_read_for_fit") is False
            and receipt.get("test_or_sample_accessed") is False,
            f"{fit_id} receipt does not bind the saved query output")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"origin", "arm", "rows", "record_ids", "cycles", "baseline",
                "head_prediction", "delta", "prediction", "missing"}
    require(required.issubset(arrays), f"{fit_id} is missing required arrays")
    require(int(arrays["origin"][0]) == origin and str(arrays["arm"][0]) == arm,
            f"{fit_id} bundle identity fields differ")
    return {"family": "RESIDUAL", "origin": origin, "arm": arm, "fit_id": fit_id,
            "path": path, "receipt_path": receipt_path, "bundle_sha256": bundle_hash,
            "receipt_file_sha256": receipt_hash, **arrays}


def selected_windows(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    cycles = np.asarray(bundle["query_cycles"] if bundle["family"] == "DIRECT" else bundle["cycles"],
                        dtype=np.int64)
    origin = int(np.asarray(bundle["origin"]).reshape(-1)[0])
    views: dict[str, np.ndarray] = {"H12": np.ones(cycles.shape, dtype=bool)} if origin == 67 else {}
    if origin == 67:
        views.update({"W4": (cycles >= 68) & (cycles <= 71),
                      "W5": (cycles >= 72) & (cycles <= 75),
                      "W6": (cycles >= 76) & (cycles <= 79)})
    else:
        views["W6"] = (cycles >= 76) & (cycles <= 79)
    return views


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    return {"mean": float(np.mean(values)), "median": float(median),
            "p10": float(q10), "p90": float(q90)}


def score_views(bundles: dict[tuple[str, int, str], dict[str, Any]],
                labels: dict[int, np.ndarray], text_load: dict[int, np.ndarray],
                direct_result: dict[str, Any], residual_scores: dict[str, Any],
                direct_freeze: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                        list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    arm_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    comparison_checks: dict[str, Any] = {}

    for (family, origin, arm), bundle in bundles.items():
        rows = np.asarray(bundle["query_rows"] if family == "DIRECT" else bundle["rows"], dtype=np.int64)
        ids = np.asarray(bundle["query_record_ids"] if family == "DIRECT" else bundle["record_ids"], dtype=str)
        cycles = np.asarray(bundle["query_cycles"] if family == "DIRECT" else bundle["cycles"], dtype=np.int64)
        baseline = np.asarray(bundle["baseline"], dtype=np.float64)
        missing = np.asarray(bundle["missing_mask"] if family == "DIRECT" else bundle["missing"], dtype=bool)
        target = np.asarray(labels[origin], dtype=np.float64)
        load = np.asarray(text_load[origin], dtype=np.float64)
        if family == "DIRECT":
            raw = np.asarray(bundle["raw_auxiliary"], dtype=np.float64)
            mapped = np.asarray(bundle["mapped_auxiliary"], dtype=np.float64)
            prediction = np.asarray(bundle["prediction"], dtype=np.float64)
            reconstructed_mapped, percentile = rank50_map(raw, cycles)
            require(np.allclose(mapped, reconstructed_mapped, rtol=0.0, atol=TOL),
                    f"{bundle['fit_id']} saved rank50 array does not reconstruct")
            reconstructed_prediction = direct_policy(baseline, mapped, missing)
            require(np.allclose(prediction, reconstructed_prediction, rtol=0.0, atol=TOL),
                    f"{bundle['fit_id']} saved policy output does not reconstruct")
            own_target = target
        else:
            raw = np.asarray(bundle["head_prediction"], dtype=np.float64)
            mapped = None
            percentile = None
            prediction = np.asarray(bundle["prediction"], dtype=np.float64)
            saved_delta = np.asarray(bundle["delta"], dtype=np.float64)
            reconstructed_delta, reconstructed_prediction = residual_policy(baseline, raw, missing)
            require(np.allclose(saved_delta, reconstructed_delta, rtol=0.0, atol=TOL)
                    and np.allclose(prediction, reconstructed_prediction, rtol=0.0, atol=TOL),
                    f"{bundle['fit_id']} saved residual correction does not reconstruct")
            own_target = target - baseline

        view_defs = selected_windows(bundle)
        for view, view_mask in view_defs.items():
            for part, part_mask in (("ALL", np.ones(rows.shape, dtype=bool)),
                                    ("PRESENT", ~missing), ("MISSING", missing)):
                mask = view_mask & part_mask
                if not np.any(mask):
                    continue
                base_mae = mae(target[mask], baseline[mask])
                final_mae = mae(target[mask], prediction[mask])
                aux_mae = mae(own_target[mask], raw[mask])
                mapped_mae = mae(own_target[mask], mapped[mask]) if mapped is not None else float("nan")
                residual = target[mask] - baseline[mask]
                correction = prediction[mask] - baseline[mask]
                direction_mask = (np.abs(residual) > 1e-12) & (np.abs(correction) > 1e-12)
                direction_match = (float(np.mean(np.sign(residual[direction_mask])
                                                 == np.sign(correction[direction_mask])))
                                   if np.any(direction_mask) else float("nan"))
                arm_rows.append({
                    "family": family, "origin": origin, "view": view, "arm": arm,
                    "subset": part, "n": int(mask.sum()), "target_scale": "threat_rank" if family == "DIRECT"
                    else "origin_specific_r06_residual", "baseline_r06_mae": base_mae,
                    "raw_auxiliary_mae_on_own_target": aux_mae,
                    "raw_auxiliary_gain_vs_reference": base_mae - aux_mae if family == "DIRECT"
                    else mae(residual, np.zeros(residual.shape, dtype=np.float64)) - aux_mae,
                    "mapped_auxiliary_mae_on_own_target": mapped_mae,
                    "final_policy_mae": final_mae, "final_policy_gain_vs_r06": base_mae - final_mae,
                    "final_policy_gain_real_vs_bag": float("nan"),
                    "final_policy_gain_real_vs_shuffled": float("nan"),
                    "correction_direction_agreement_vs_r06_residual": direction_match,
                    "direction_compared_rows": int(direction_mask.sum()),
                })

            # Fixed per-cycle auxiliary correlations and final policy signs.
            for cycle in sorted(np.unique(cycles[view_mask]).tolist()):
                cycle_mask = view_mask & (cycles == cycle)
                for subset, submask in (("ALL", cycle_mask), ("PRESENT", cycle_mask & ~missing)):
                    rho = spearman(own_target[submask], raw[submask])
                    if not np.isfinite(rho):
                        continue
                    base_cycle_mae = mae(target[submask], baseline[submask])
                    aux_cycle_mae = mae(own_target[submask], raw[submask])
                    final_cycle_mae = mae(target[submask], prediction[submask])
                    cycle_rows.append({
                        "family": family, "origin": origin, "view": view, "arm": arm,
                        "cycle": int(cycle), "subset": subset, "n": int(submask.sum()),
                        "auxiliary_target": "threat_rank" if family == "DIRECT" else "target_minus_origin_r06",
                        "auxiliary_mae": aux_cycle_mae,
                        "auxiliary_gain_vs_reference": base_cycle_mae - aux_cycle_mae if family == "DIRECT"
                        else mae(target[submask] - baseline[submask], np.zeros(int(submask.sum()))) - aux_cycle_mae,
                        "within_cycle_spearman": rho,
                        "final_policy_mae": final_cycle_mae,
                        "final_policy_gain_vs_r06": base_cycle_mae - final_cycle_mae,
                    })

            # Direct-only rank50 change is decomposed before the frozen missing fallback.
            if family == "DIRECT":
                rank_shift = 0.5 * (percentile - raw)
                for part, part_mask in (("ALL", np.ones(rows.shape, dtype=bool)),
                                        ("PRESENT", ~missing), ("MISSING", missing)):
                    mask = view_mask & part_mask
                    if not np.any(mask):
                        continue
                    shift_stats = summarize_values(rank_shift[mask])
                    residual = target[mask] - baseline[mask]
                    shift_direction = (np.abs(residual) > 1e-12) & (np.abs(rank_shift[mask]) > 1e-12)
                    correction = prediction[mask] - baseline[mask]
                    policy_direction = (np.abs(residual) > 1e-12) & (np.abs(correction) > 1e-12)
                    rank_rows.append({
                        "origin": origin, "view": view, "arm": arm, "subset": part,
                        "n": int(mask.sum()),
                        "raw_auxiliary_mae": mae(target[mask], raw[mask]),
                        "mapped_auxiliary_mae": mae(target[mask], mapped[mask]),
                        "raw_to_mapped_auxiliary_mae_gain": mae(target[mask], raw[mask]) - mae(target[mask], mapped[mask]),
                        "rank_shift_mean": shift_stats["mean"], "rank_shift_median": shift_stats["median"],
                        "rank_shift_p10": shift_stats["p10"], "rank_shift_p90": shift_stats["p90"],
                        "rank_shift_sign_agreement_with_r06_residual": float(np.mean(
                            np.sign(residual[shift_direction]) == np.sign(rank_shift[mask][shift_direction])))
                            if np.any(shift_direction) else float("nan"),
                        "rank_shift_direction_rows": int(shift_direction.sum()),
                        "policy_correction_mean": float(np.mean(correction)),
                        "policy_correction_median": float(np.median(correction)),
                        "policy_correction_p10": float(np.quantile(correction, 0.1)),
                        "policy_correction_p90": float(np.quantile(correction, 0.9)),
                        "policy_correction_sign_agreement_with_r06_residual": float(np.mean(
                            np.sign(residual[policy_direction]) == np.sign(correction[policy_direction])))
                            if np.any(policy_direction) else float("nan"),
                        "policy_correction_direction_rows": int(policy_direction.sum()),
                        "baseline_r06_mae": mae(target[mask], baseline[mask]),
                        "final_policy_mae": mae(target[mask], prediction[mask]),
                        "final_policy_gain_vs_r06": mae(target[mask], baseline[mask]) - mae(target[mask], prediction[mask]),
                    })

            # Frozen target tails and prefit text-load quartiles; small cells stay UNKNOWN.
            group_masks: dict[str, np.ndarray] = {
                "TARGET_LE5": target <= 5.0,
                "TARGET_GT5_LT95": (target > 5.0) & (target < 95.0),
                "TARGET_GE95": target >= 95.0,
            }
            edges = np.asarray(direct_freeze["origins"][str(origin)]["group_edges"]["text_load"], dtype=np.float64)
            text_q = np.digitize(load, edges, right=False)
            for q in range(len(edges) + 1):
                group_masks[f"TEXT_LOAD_Q{q + 1}"] = text_q == q
            for group_name, group_mask in group_masks.items():
                mask = view_mask & group_mask
                n = int(mask.sum())
                row = {"family": family, "origin": origin, "view": view, "arm": arm,
                       "group": group_name, "n": n,
                       "status": "SUPPORTED" if n >= MIN_GROUP_N else "UNKNOWN_LT_100",
                       "target_scale": "threat_rank" if family == "DIRECT" else "origin_specific_r06_residual"}
                if n >= MIN_GROUP_N:
                    row.update({
                        "r06_mae": mae(target[mask], baseline[mask]),
                        "raw_auxiliary_mae_on_own_target": mae(own_target[mask], raw[mask]),
                        "final_policy_mae": mae(target[mask], prediction[mask]),
                        "final_policy_gain_vs_r06": mae(target[mask], baseline[mask])
                        - mae(target[mask], prediction[mask]),
                    })
                else:
                    row.update({"r06_mae": "", "raw_auxiliary_mae_on_own_target": "",
                                "final_policy_mae": "", "final_policy_gain_vs_r06": ""})
                group_rows.append(row)

    # Add same-window REAL control contrasts to the long arm table.
    index = {(r["family"], r["origin"], r["view"], r["subset"], r["arm"]): r for r in arm_rows}
    for row in arm_rows:
        real = index[(row["family"], row["origin"], row["view"], row["subset"], "REAL")]
        bag = index[(row["family"], row["origin"], row["view"], row["subset"], "BAG")]
        shuffled = index[(row["family"], row["origin"], row["view"], row["subset"], "SHUFFLED")]
        row["final_policy_gain_real_vs_bag"] = bag["final_policy_mae"] - real["final_policy_mae"]
        row["final_policy_gain_real_vs_shuffled"] = shuffled["final_policy_mae"] - real["final_policy_mae"]

    # Reproduce the frozen aggregate score tables from the saved prediction bundles.
    for family, saved_scores in (("DIRECT", direct_result), ("RESIDUAL", residual_scores)):
        for origin in ORIGINS:
            for view, view_mask in selected_windows(bundles[(family, origin, "BAG")]).items():
                saved_view = "E67_H12" if origin == 67 and view == "H12" else (
                    f"E{origin}_{view}" if origin == 67 else "E75_W6")
                metrics = saved_scores["views"][saved_view]["metrics"]
                bag = bundles[(family, origin, "BAG")]
                rows = np.asarray(bag["query_rows"] if family == "DIRECT" else bag["rows"], dtype=np.int64)
                baseline = np.asarray(bag["baseline"], dtype=np.float64)
                target = labels[origin]
                check_values = {"R06.mae": (metrics["R06"]["mae"], mae(target[view_mask], baseline[view_mask]))}
                for arm in ARMS:
                    b = bundles[(family, origin, arm)]
                    prediction = np.asarray(b["prediction"], dtype=np.float64)
                    saved_arm = metrics[arm]
                    check_values[f"{arm}.mae"] = (saved_arm["mae"], mae(target[view_mask], prediction[view_mask]))
                    check_values[f"{arm}.gain_vs_R06"] = (saved_arm["gain_vs_R06"],
                                                            mae(target[view_mask], baseline[view_mask])
                                                            - mae(target[view_mask], prediction[view_mask]))
                for metric_name, (saved, computed) in check_values.items():
                    require(np.isclose(saved, computed, rtol=0.0, atol=TOL),
                            f"Saved {family} {saved_view} {metric_name} does not match bundle reconstruction")
                comparison_checks[f"{family}_{saved_view}"] = {k: {"saved": float(v[0]), "recomputed": float(v[1]),
                                                                     "match": True}
                                                               for k, v in check_values.items()}
    return arm_rows, cycle_rows, rank_rows, group_rows, comparison_checks


def matched_w6(bundles: dict[tuple[str, int, str], dict[str, Any]],
               labels: dict[int, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    cycles_out: list[dict[str, Any]] = []
    for family in ("DIRECT", "RESIDUAL"):
        for arm in ARMS:
            by_origin = {}
            for origin in ORIGINS:
                b = bundles[(family, origin, arm)]
                cycles = np.asarray(b["query_cycles"] if family == "DIRECT" else b["cycles"], dtype=np.int64)
                rows = np.asarray(b["query_rows"] if family == "DIRECT" else b["rows"], dtype=np.int64)
                mask = (cycles >= 76) & (cycles <= 79)
                target = labels[origin][mask]
                baseline = np.asarray(b["baseline"], dtype=np.float64)[mask]
                raw = np.asarray(b["raw_auxiliary"] if family == "DIRECT" else b["head_prediction"], dtype=np.float64)[mask]
                final = np.asarray(b["prediction"], dtype=np.float64)[mask]
                raw_target = target if family == "DIRECT" else target - baseline
                raw_mae = mae(raw_target, raw)
                reference_mae = mae(target, baseline) if family == "DIRECT" else mae(target - baseline, np.zeros(target.shape))
                policy_mae = mae(target, final)
                metrics = {"origin": origin, "n": int(mask.sum()), "r06_mae": mae(target, baseline),
                           "final_policy_mae": policy_mae, "final_policy_gain_vs_r06": mae(target, baseline) - policy_mae,
                           "raw_auxiliary_target": "threat_rank" if family == "DIRECT" else "target_minus_origin_r06",
                           "raw_auxiliary_mae": raw_mae, "raw_auxiliary_reference_mae": reference_mae,
                           "raw_auxiliary_gain_vs_reference": reference_mae - raw_mae}
                by_origin[origin] = metrics
                for cycle in range(76, 80):
                    cm = mask & (cycles == cycle)
                    yt = labels[origin][cm]
                    base = np.asarray(b["baseline"], dtype=np.float64)[cm]
                    aux = np.asarray(b["raw_auxiliary"] if family == "DIRECT" else b["head_prediction"], dtype=np.float64)[cm]
                    pred = np.asarray(b["prediction"], dtype=np.float64)[cm]
                    aux_target = yt if family == "DIRECT" else yt - base
                    base_aux_mae = mae(yt, base) if family == "DIRECT" else mae(yt - base, np.zeros(yt.shape))
                    aux_mae = mae(aux_target, aux)
                    cycles_out.append({"family": family, "arm": arm, "origin": origin, "cycle": cycle,
                                       "n": int(cm.sum()), "r06_mae": mae(yt, base),
                                       "auxiliary_target": metrics["raw_auxiliary_target"],
                                       "raw_auxiliary_mae": aux_mae,
                                       "raw_auxiliary_gain_vs_reference": base_aux_mae - aux_mae,
                                       "final_policy_mae": mae(yt, pred),
                                       "final_policy_gain_vs_r06": mae(yt, base) - mae(yt, pred)})
            e67, e75 = by_origin[67], by_origin[75]
            summary.append({
                "family": family, "arm": arm, "matched_w6_n": e67["n"],
                "same_ids_and_targets_verified": True,
                "r06_mae_e67": e67["r06_mae"], "r06_mae_e75": e75["r06_mae"],
                "r06_improvement_e67_to_e75": e67["r06_mae"] - e75["r06_mae"],
                "final_policy_mae_e67": e67["final_policy_mae"], "final_policy_mae_e75": e75["final_policy_mae"],
                "final_policy_gain_vs_r06_e67": e67["final_policy_gain_vs_r06"],
                "final_policy_gain_vs_r06_e75": e75["final_policy_gain_vs_r06"],
                "raw_auxiliary_target": e67["raw_auxiliary_target"],
                "raw_auxiliary_mae_e67": e67["raw_auxiliary_mae"],
                "raw_auxiliary_mae_e75": e75["raw_auxiliary_mae"],
                "raw_auxiliary_reference_mae_e67": e67["raw_auxiliary_reference_mae"],
                "raw_auxiliary_reference_mae_e75": e75["raw_auxiliary_reference_mae"],
                "raw_auxiliary_gain_vs_reference_e67": e67["raw_auxiliary_gain_vs_reference"],
                "raw_auxiliary_gain_vs_reference_e75": e75["raw_auxiliary_gain_vs_reference"],
                "residual_raw_loss_cross_origin_comparable": "NO_DIFFERENT_R06_RESIDUAL_TARGETS"
                if family == "RESIDUAL" else "YES_SAME_THREAT_RANK_TARGET",
            })
    # Explicit per-cycle E75-minus-E67 current baseline movement on the same rows.
    for arm in ARMS:
        b67 = bundles[("DIRECT", 67, arm)]
        b75 = bundles[("DIRECT", 75, arm)]
        cy67 = np.asarray(b67["query_cycles"], dtype=np.int64)
        cy75 = np.asarray(b75["query_cycles"], dtype=np.int64)
        ids67 = np.asarray(b67["query_record_ids"], dtype=str)
        ids75 = np.asarray(b75["query_record_ids"], dtype=str)
        map67 = {str(record_id): i for i, record_id in enumerate(ids67)}
        map75 = {str(record_id): i for i, record_id in enumerate(ids75)}
        ids = sorted(set(map67).intersection(map75), key=lambda x: map67[x])
        require(len(ids) == 8491, "Matched W6 record_id intersection changed")
        i67 = np.asarray([map67[x] for x in ids], dtype=np.int64)
        i75 = np.asarray([map75[x] for x in ids], dtype=np.int64)
        require(np.array_equal(cy67[i67], cy75[i75]) and np.array_equal(labels[67][i67], labels[75][i75]),
                "Matched W6 E67/E75 cycles or query labels differ")
    # Put the baseline catch-up and auxiliary change beside each per-cycle sign.
    cycle_index = {(x["family"], x["arm"], x["origin"], x["cycle"]): x for x in cycles_out}
    for family in ("DIRECT", "RESIDUAL"):
        for arm in ARMS:
            for cycle in range(76, 80):
                e67 = cycle_index[(family, arm, 67, cycle)]
                e75 = cycle_index[(family, arm, 75, cycle)]
                baseline_improvement = e67["r06_mae"] - e75["r06_mae"]
                for item in (e67, e75):
                    item["r06_improvement_e67_to_e75_same_cycle"] = baseline_improvement
                if family == "DIRECT":
                    aux_change = e67["raw_auxiliary_mae"] - e75["raw_auxiliary_mae"]
                    e67["direct_auxiliary_mae_improvement_e67_to_e75"] = aux_change
                    e75["direct_auxiliary_mae_improvement_e67_to_e75"] = aux_change
                else:
                    e67["residual_auxiliary_cross_origin_change"] = "UNKNOWN_DIFFERENT_RESIDUAL_TARGETS"
                    e75["residual_auxiliary_cross_origin_change"] = "UNKNOWN_DIFFERENT_RESIDUAL_TARGETS"
    return summary, cycles_out


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(repo_root: Path, source_root: Path, saved_root: Path, output_root: Path) -> dict[str, Any]:
    repo_root, source_root, saved_root, output_root = (p.resolve() for p in
                                                        (repo_root, source_root, saved_root, output_root))
    expected_output = (repo_root / OUTPUT_REL).resolve()
    require(output_root == expected_output, "Output must be the dedicated failure-mechanism report directory")
    require(repo_root not in source_root.parents and source_root not in repo_root.parents,
            "Source root and report worktree must be separate")
    require(saved_root != repo_root, "Saved pilot artifacts must come from the preserved source worktree")

    # Read scored reports from their preserved source worktree. This keeps
    # byte hashes stable across Windows checkout line-ending normalization.
    direct_report = saved_root / DIRECT_REPORT_REL
    residual_report = saved_root / RESIDUAL_REPORT_REL
    transport_report = saved_root / TRANSPORT_REL
    direct_run = saved_root / DIRECT_RUN_REL
    residual_run = saved_root / RESIDUAL_RUN_REL
    direct_freeze = read_json(direct_run / "FREEZE_V4.json")
    direct_result = read_json(direct_report / "RESULTS.json")
    direct_completion = read_json(direct_report / "CONTINUATION_COMPLETION_V1.json")
    direct_prelabel = read_json(direct_report / "CONTINUATION_PRELABEL_VERIFICATION_V1.json")
    direct_final = read_json(direct_report / "CONTINUATION_FINAL_VERIFICATION_V1.json")
    direct_recipe = read_json(direct_report / "RECIPE.json")
    verify_body_signature(direct_freeze, "freeze_signature", "Direct V4 freeze")
    verify_body_signature(direct_completion, "completion_signature", "Direct completion")
    verify_body_signature(direct_prelabel, "verification_signature", "Direct prelabel")
    verify_body_signature(direct_final, "verification_signature", "Direct final verification")
    require(direct_freeze["freeze_signature"] == DIRECT_FREEZE_SIGNATURE
            and direct_recipe["card_id"] == DIRECT_CARD
            and direct_freeze["recipe_sha256"] == sha_file(direct_report / "RECIPE.json"),
            "Direct V4 freeze or recipe identity changed")
    direct_policy = direct_recipe["fixed_competition_policy"]
    require("0.5*raw+0.5*percentile" in direct_policy["rank_mapping"]
            and "0.75*cached_current_R06 + 0.25*mapped_auxiliary" in direct_policy["blend"]
            and int(direct_recipe["gates"]["protected_groups"]["minimum_n"]) == MIN_GROUP_N,
            "Direct rank50/blend arithmetic differs from its frozen recipe")
    require(direct_completion["status"] == "PASS" and direct_completion["fit_units_started"] == 12
            and direct_completion["fit_units_saved"] == 12 and direct_completion["fit_units_failed"] == 0
            and direct_completion["remaining_fit_units"] == 0 and direct_completion["query_targets_read"] is False,
            "Direct completion record does not preserve 12/12 and prelabel target boundary")
    require(direct_prelabel["status"] == "PASS" and direct_prelabel["fit_units_saved"] == 12
            and direct_prelabel["query_targets_read"] is False and direct_final["verification_status"] == "PASS"
            and direct_final["candidate_status"] == "FAIL" and direct_result["status"] == "FAIL",
            "Direct prelabel/final decision records disagree")
    require(direct_final["results_sha256"] == sha_file(direct_report / "RESULTS.json")
            and direct_final["report_sha256"] == sha_file(direct_report / "REPORT.md"),
            "Direct final verification no longer binds the scored result/report")

    residual_freeze = read_json(residual_report / "FREEZE.json")
    residual_recipe = read_json(residual_report / "RECIPE.json")
    residual_score = read_json(residual_report / "SCORE_METRICS.json")
    transport_inputs = read_json(transport_report / "INPUT_HASHES.json")
    legacy_hashes = transport_inputs["pilot_artifact_hashes_before"]
    verify_body_signature(residual_freeze, "freeze_signature", "Residual pilot freeze")
    require(residual_freeze["freeze_signature"] == RESIDUAL_FREEZE_SIGNATURE
            and residual_recipe["card_id"] == RESIDUAL_CARD
            and float(residual_freeze["policy"]["alpha"]) == 0.25
            and float(residual_freeze["policy"]["delta_cap"]) == 5.0
            and residual_score["freeze_signature"] == RESIDUAL_FREEZE_SIGNATURE
            and residual_score["status"] == "SCORED_AFTER_PRELABEL_PASS"
            and residual_score["fit_starts"] == 6 and residual_score["fit_saved"] == 6
            and residual_score["fit_failed"] == 0
            and residual_score["query_targets_read_after_all_six_saved"] is True,
            "Residual pilot freeze or post-prelabel scoring boundary failed")
    for legacy_key, path in (("FREEZE", residual_report / "FREEZE.json"),
                             ("PRELABEL", residual_report / "PRELABEL_VERIFICATION.json"),
                             ("SCORE", residual_report / "SCORE_METRICS.json")):
        require(legacy_hashes.get(legacy_key) == sha_file(path),
                f"Prior transport audit hash changed for residual {legacy_key}")
    require(transport_inputs["freeze_signature"] == RESIDUAL_FREEZE_SIGNATURE
            and transport_inputs["pilot_ledger_sha256"] == legacy_hashes["LEDGER"],
            "Prior zero-fit residual transport audit lineage changed")

    bundles: dict[tuple[str, int, str], dict[str, Any]] = {}
    for origin in ORIGINS:
        for arm in ARMS:
            d = load_direct_bundle(saved_root, direct_report, direct_freeze, direct_prelabel, origin, arm)
            r = load_residual_bundle(saved_root, legacy_hashes, residual_freeze, origin, arm)
            bundles[("DIRECT", origin, arm)] = d
            bundles[("RESIDUAL", origin, arm)] = r

    # Select exactly the already-scored E67 query rows (cycles 68-79).
    direct67 = bundles[("DIRECT", 67, "BAG")]
    query_rows = np.asarray(direct67["query_rows"], dtype=np.int64)
    query_ids = np.asarray(direct67["query_record_ids"], dtype=str)
    query_cycles = np.asarray(direct67["query_cycles"], dtype=np.int64)
    require(query_rows.size == 20_903 and np.all(query_cycles > 67),
            "E67 saved query boundary does not match the previously scored view")
    targets_all, text_load_all, parsed_rows, source_audit = read_scored_query(
        source_root, query_rows, query_ids, query_cycles, expected_total_rows=50_608)
    require(source_audit["query_target_values_sha256"] == EXPECTED_DIRECT_TARGET_SHA256
            and direct_result["query_target_rows"] == len(targets_all)
            and direct_result["query_targets_sha256"] == EXPECTED_DIRECT_TARGET_SHA256,
            "Previously opened target vector does not match the scored direct pilot")
    target_by_row = {int(row): float(value) for row, value in zip(parsed_rows, targets_all)}
    load_by_row = {int(row): float(value) for row, value in zip(parsed_rows, text_load_all)}
    labels: dict[int, np.ndarray] = {}
    text_load: dict[int, np.ndarray] = {}
    for origin in ORIGINS:
        b = bundles[("DIRECT", origin, "BAG")]
        rows = np.asarray(b["query_rows"], dtype=np.int64)
        ids = np.asarray(b["query_record_ids"], dtype=str)
        cycles = np.asarray(b["query_cycles"], dtype=np.int64)
        labels[origin] = np.asarray([target_by_row[int(row)] for row in rows], dtype=np.float64)
        text_load[origin] = np.asarray([load_by_row[int(row)] for row in rows], dtype=np.float64)
        for family in ("DIRECT", "RESIDUAL"):
            for arm in ARMS:
                bundle = bundles[(family, origin, arm)]
                b_rows = np.asarray(bundle["query_rows"] if family == "DIRECT" else bundle["rows"], dtype=np.int64)
                b_ids = np.asarray(bundle["query_record_ids"] if family == "DIRECT" else bundle["record_ids"], dtype=str)
                b_cycles = np.asarray(bundle["query_cycles"] if family == "DIRECT" else bundle["cycles"], dtype=np.int64)
                verify_alignment(b_rows, b_ids, b_cycles, rows, ids, cycles, bundle["fit_id"])
                require(np.isfinite(bundle["baseline"]).all() and np.array_equal(bundle["baseline"],
                        bundles[("DIRECT", origin, "BAG")]["baseline"]),
                        f"{bundle['fit_id']} baseline differs from the same-origin direct R06 baseline")
        for arm in ARMS:
            d = bundles[("DIRECT", origin, arm)]
            r = bundles[("RESIDUAL", origin, arm)]
            require(np.array_equal(d["missing_mask"], r["missing"]),
                    f"{origin} {arm} direct/residual missing masks differ")

    # The E67 and E75 W6s are the same official record IDs, cycles, targets and text-load values.
    rows67 = np.asarray(bundles[("DIRECT", 67, "BAG")]["query_rows"], dtype=np.int64)
    rows75 = np.asarray(bundles[("DIRECT", 75, "BAG")]["query_rows"], dtype=np.int64)
    mask67 = np.isin(np.asarray(bundles[("DIRECT", 67, "BAG")]["query_cycles"], dtype=np.int64), [76, 77, 78, 79])
    mask75 = np.isin(np.asarray(bundles[("DIRECT", 75, "BAG")]["query_cycles"], dtype=np.int64), [76, 77, 78, 79])
    require(np.array_equal(rows67[mask67], rows75[mask75])
            and np.array_equal(labels[67][mask67], labels[75][mask75])
            and np.array_equal(text_load[67][mask67], text_load[75][mask75]),
            "E67/E75 W6 is not the same 8,491 row/target/text-load population")
    require(rows67[mask67].size == 8_491, "Matched W6 denominator changed")

    arm_rows, cycle_rows, rank_rows, group_rows, comparison_checks = score_views(
        bundles, labels, text_load, direct_result, residual_score, direct_freeze)
    w6_summary, w6_cycles = matched_w6(bundles, labels)

    def arm_metric(family: str, origin: int, view: str, arm: str) -> dict[str, Any]:
        matches = [x for x in arm_rows if x["family"] == family and x["origin"] == origin
                   and x["view"] == view and x["arm"] == arm and x["subset"] == "ALL"]
        require(len(matches) == 1, f"Missing/duplicate metric row: {family} E{origin} {view} {arm}")
        return matches[0]

    def rank_metric(origin: int, view: str, arm: str) -> dict[str, Any]:
        matches = [x for x in rank_rows if x["origin"] == origin and x["view"] == view
                   and x["arm"] == arm and x["subset"] == "ALL"]
        require(len(matches) == 1, f"Missing/duplicate rank row: E{origin} {view} {arm}")
        return matches[0]

    d67, d75 = arm_metric("DIRECT", 67, "W6", "REAL"), arm_metric("DIRECT", 75, "W6", "REAL")
    r67, r75 = rank_metric(67, "W6", "REAL"), rank_metric(75, "W6", "REAL")
    d67_controls = {a: arm_metric("DIRECT", 67, "W6", a) for a in ARMS}
    d75_controls = {a: arm_metric("DIRECT", 75, "W6", a) for a in ARMS}
    z67, z75 = arm_metric("RESIDUAL", 67, "W6", "REAL"), arm_metric("RESIDUAL", 75, "W6", "REAL")
    mechanism_assessment = {
        "matched_w6_n": 8_491,
        "direct_real": {
            "r06_mae_e67": d67["baseline_r06_mae"], "r06_mae_e75": d75["baseline_r06_mae"],
            "r06_mae_improvement_e67_to_e75": d67["baseline_r06_mae"] - d75["baseline_r06_mae"],
            "raw_aux_mae_e67": d67["raw_auxiliary_mae_on_own_target"],
            "raw_aux_mae_e75": d75["raw_auxiliary_mae_on_own_target"],
            "raw_aux_absolute_mae_improvement_e67_to_e75":
                d67["raw_auxiliary_mae_on_own_target"] - d75["raw_auxiliary_mae_on_own_target"],
            "raw_aux_gain_vs_r06_e67": d67["raw_auxiliary_gain_vs_reference"],
            "raw_aux_gain_vs_r06_e75": d75["raw_auxiliary_gain_vs_reference"],
            "relative_auxiliary_gap_change_e75_minus_e67":
                d75["raw_auxiliary_gain_vs_reference"] - d67["raw_auxiliary_gain_vs_reference"],
            "final_policy_gain_vs_r06_e67": d67["final_policy_gain_vs_r06"],
            "final_policy_gain_vs_r06_e75": d75["final_policy_gain_vs_r06"],
            "raw_to_rank50_auxiliary_mae_gain_e67": r67["raw_to_mapped_auxiliary_mae_gain"],
            "raw_to_rank50_auxiliary_mae_gain_e75": r75["raw_to_mapped_auxiliary_mae_gain"],
            "rank50_auxiliary_gain_vs_r06_e67": d67["baseline_r06_mae"] - r67["mapped_auxiliary_mae"],
            "rank50_auxiliary_gain_vs_r06_e75": d75["baseline_r06_mae"] - r75["mapped_auxiliary_mae"],
            "policy_correction_direction_agreement_e67": r67["policy_correction_sign_agreement_with_r06_residual"],
            "policy_correction_direction_agreement_e75": r75["policy_correction_sign_agreement_with_r06_residual"],
            "real_gain_vs_bag_e67": d67_controls["REAL"]["final_policy_gain_real_vs_bag"],
            "real_gain_vs_shuffled_e67": d67_controls["REAL"]["final_policy_gain_real_vs_shuffled"],
            "real_gain_vs_bag_e75": d75_controls["REAL"]["final_policy_gain_real_vs_bag"],
            "real_gain_vs_shuffled_e75": d75_controls["REAL"]["final_policy_gain_real_vs_shuffled"],
            "all_e75_arms_harm_r06": all(d75_controls[a]["final_policy_gain_vs_r06"] < 0.0 for a in ARMS),
        },
        "prior_residual_real": {
            "target_scale": "origin_specific_threat_rank_minus_r06",
            "raw_aux_gain_vs_zero_residual_e67": z67["raw_auxiliary_gain_vs_reference"],
            "raw_aux_gain_vs_zero_residual_e75": z75["raw_auxiliary_gain_vs_reference"],
            "final_policy_gain_vs_r06_e67": z67["final_policy_gain_vs_r06"],
            "final_policy_gain_vs_r06_e75": z75["final_policy_gain_vs_r06"],
            "raw_auxiliary_loss_comparable_across_origins": False,
        },
        "interpretation": "On matched rows, direct REAL raw auxiliary MAE improves in absolute terms from E67 to E75, but R06 improves more, so its relative auxiliary deficit widens. Rank50 reduces direct auxiliary MAE, especially at E67, yet the mapped head remains worse than R06 and E75's saved 25% policy harms all three arms. REAL-versus-control contrasts change sign, but this is a secondary descriptive contrast and does not isolate a positional-information cause.",
    }

    # Confirm the fixed text-load bins are the already-frozen fit-prefix quartiles.
    group_edges = {str(origin): direct_freeze["origins"][str(origin)]["group_edges"]["text_load"]
                   for origin in ORIGINS}
    require(all(len(v) == 3 for v in group_edges.values()), "Frozen text-load quartile edges are incomplete")

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "arm_window_metrics.csv": arm_rows,
        "within_cycle_metrics.csv": cycle_rows,
        "direct_rank50_decomposition.csv": rank_rows,
        "matched_w6_origin_changes.csv": w6_summary,
        "matched_w6_by_cycle.csv": w6_cycles,
        "fixed_group_metrics.csv": group_rows,
    }
    output_hashes: dict[str, str] = {}
    for name, rows in outputs.items():
        payload = csv_bytes(rows)
        path = output_root / name
        path.write_bytes(payload)
        output_hashes[name] = hashlib.sha256(payload).hexdigest()

    inputs: dict[str, str] = {}
    input_paths = [
        repo_root / "scripts/reg_position_failure_mechanism_v1.py",
        direct_report / "RECIPE.json", direct_report / "RESULTS.json", direct_report / "REPORT.md",
        direct_report / "CONTINUATION_COMPLETION_V1.json",
        direct_report / "CONTINUATION_PRELABEL_VERIFICATION_V1.json",
        direct_report / "CONTINUATION_FINAL_VERIFICATION_V1.json",
        direct_report / "PRELABEL_VERIFICATION.json", direct_run / "FREEZE_V4.json",
        residual_report / "RECIPE.json", residual_report / "FREEZE.json",
        residual_report / "FINDINGS.md", residual_report / "SCORE_METRICS.json",
        residual_report / "PRELABEL_VERIFICATION.json",
        transport_report / "TRANSPORT_DIAGNOSTIC.md", transport_report / "INPUT_HASHES.json",
        repo_root / "reports/optimization_v3/supervised_pilot/FULLPILOT.md",
        repo_root / "reports/optimization_v3/supervised_pilot/verification.json",
        repo_root / "reports/optimization_v3/batch1/R07/summary.md",
    ]
    for path in input_paths:
        require(path.is_file(), f"Required provenance input is missing: {path}")
        try:
            key = path.relative_to(repo_root).as_posix()
        except ValueError:
            key = "saved_artifact_root/" + path.relative_to(saved_root).as_posix()
        inputs[key] = sha_file(path)
    for (family, origin, arm), b in bundles.items():
        prefix = f"{family}_E{origin}_{arm}"
        inputs[f"saved_bundles/{prefix}.npz"] = b["bundle_sha256"]
        inputs[f"saved_bundles/{prefix}.receipt"] = b["receipt_file_sha256"]

    evidence = {
        "schema_version": 1,
        "audit_id": "REG_POSITION_FAILURE_MECHANISM_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ZERO_FIT_DIAGNOSTIC",
        "decision_preserved": "REG_POSITION_DIRECT_L1_V1_FAIL_NO_PROMOTION",
        "source_boundary": {
            **source_audit,
            "source_root": str(source_root),
            "saved_artifact_root": str(saved_root),
            "direct_freeze_signature": direct_freeze["freeze_signature"],
            "direct_continuation_signature": direct_completion["continuation_signature"],
            "direct_repair_signature": direct_completion["postfit_validator_repair"]["repair_signature"],
            "residual_freeze_signature": residual_freeze["freeze_signature"],
            "target_values_are_only_previously_scored_queries": True,
            "test_sample_kaggle_accessed": False,
            "fit_or_predict_called": False,
        },
        "population": {
            "e67_query_rows": 20_903, "e75_query_rows": 8_491,
            "matched_w6_rows": 8_491, "matched_w6_ids_targets_cycles_text_load_exact": True,
            "adaptive_repeated_development_use": True,
            "e67_h12_contains_e75_w6": True,
            "untouched_validation": False,
            "text_load_edges_from_direct_v4_fit_prefix": group_edges,
            "fixed_min_group_n": MIN_GROUP_N,
            "target_tails": ["target<=5", "5<target<95", "target>=95"],
        },
        "comparison_checks": comparison_checks,
        "mechanism_assessment": mechanism_assessment,
        "historical_novelty": {
            "rank50_fixed_blend_previously_executed": True,
            "evidence": [
                "supervised_pilot/FULLPILOT.md reports raw, within-cycle rank50 and 75/25 champion/rank50 regression; its independent verification is PASS, but its fixed-blend gate FAILs (primary gain -0.116570, H12 gain -0.128519).",
                "R07 summary reports the same 75/25 champion plus within-cycle rank50 family and retires it after negative primary/H12 gains.",
                "REG_POSITION_TRANSPORT_DIAGNOSTIC_V1 already compared residual-position saved arms on the exact paired W6 IDs, baseline changes, raw head/delta summaries, per-cycle metrics and residual distributions.",
            ],
            "novelty_conclusion": "The broad rank50+blend and residual-position transport ideas are not novel here. Direct-target rank50 decomposition is a new diagnostic of the frozen failed pilot, not evidence for another candidate.",
            "next_lane": "NONE_JUSTIFIED",
            "reason": "The frozen direct pilot fails on E75 W6 against R06, BAG and SHUFFLED; prior executed rank50 blends and residual-position transport diagnostics cover the nearest candidate families. This post-hoc decomposition cannot isolate a causal positional-information effect. No new fit or recipe lane is justified by this audit.",
        },
        "outputs": output_hashes,
        "inputs": inputs,
    }
    write_json(output_root / "AUDIT.json", evidence)
    write_json(output_root / "INPUT_HASHES.json", {"script_sha256": inputs[
        "scripts/reg_position_failure_mechanism_v1.py"], "input_hashes": inputs,
        "output_hashes_before_report": output_hashes})

    lines = [
        "# REG_POSITION_FAILURE_MECHANISM_V1",
        "",
        "**Status: PASS — zero-fit descriptive audit.** This is adaptive repeated-development evidence, not untouched validation. E67_H12 contains the E75_W6 records. The REG_POSITION_DIRECT_L1_V1 FAIL verdict is preserved. This report diagnoses saved outputs; it does not change selection, score a new policy, or establish causality.",
        "",
        f"Created UTC: {evidence['created_utc']}",
        f"Frozen direct signature: `{direct_freeze['freeze_signature']}`; continuation `{direct_completion['continuation_signature']}`; repair `{direct_completion['postfit_validator_repair']['repair_signature']}`.",
        f"Query boundary: {source_audit['query_target_rows_parsed']:,} already-scored labels, cycles {source_audit['selected_cycle_min']}-{source_audit['selected_cycle_max']}; zero fit-prefix target rows parsed. Test, Sample and Kaggle were not accessed.",
        "",
        "## Frozen score verification",
        "",
        f"Recomputed `{len(comparison_checks)}` origin/view score blocks from saved prediction bundles. Every saved R06, arm MAE, and arm gain reproduced within {TOL:g}.",
        "",
        "## Direct versus residual target scales",
        "",
        "`arm_window_metrics.csv` reports direct auxiliary MAE against `threat_rank`. Residual auxiliary MAE uses the correct origin-specific target `threat_rank - that origin's saved R06`; those residual losses are not compared across origins or against direct target-scale losses. Final policy MAE is always evaluated against the same official threat_rank labels.",
        "",
        "The table includes each arm's within-cycle Spearman (summarized by unweighted cycle mean/median in `within_cycle_metrics.csv`), final policy MAE, R06 gain and REAL-vs-control differences. H12 and W4/W5/W6 use the frozen cycle windows.",
        "",
        "## Rank50 and matched W6 decomposition",
        "",
        "`direct_rank50_decomposition.csv` recomputes the saved full-cycle map before the missing fallback. Rank shift is exactly `0.5 * (percentile - raw)`; it reports signed mean/median/p10/p90, raw/mapped auxiliary MAE, rank-shift direction versus the R06 residual, and the actual frozen policy correction direction versus that residual. Present and missing rows are separate; missing policy outputs remain exact R06.",
        "",
        "`matched_w6_origin_changes.csv` separates current R06 improvement from auxiliary quality and final policy gain on the same 8,491 IDs/cycles/labels. Direct raw auxiliary scores use threat_rank. Residual heads use origin-specific residual targets and compare only to that origin's zero-residual reference. Per-cycle signs and same-cycle R06 catch-up are in `matched_w6_by_cycle.csv`.",
        "",
        "### What the saved outputs support",
        "",
        f"On matched W6, R06 MAE improves by {mechanism_assessment['direct_real']['r06_mae_improvement_e67_to_e75']:.6f} from E67 to E75. Direct REAL raw-head MAE improves by {mechanism_assessment['direct_real']['raw_aux_absolute_mae_improvement_e67_to_e75']:.6f}, so its raw-head deficit versus R06 widens by {abs(mechanism_assessment['direct_real']['relative_auxiliary_gap_change_e75_minus_e67']):.6f}. This is evidence of baseline catch-up relative to direct auxiliary quality; the absolute head loss itself improves.",
        f"The existing rank50 map lowers direct REAL auxiliary MAE by {mechanism_assessment['direct_real']['raw_to_rank50_auxiliary_mae_gain_e67']:.6f} at E67 W6 and {mechanism_assessment['direct_real']['raw_to_rank50_auxiliary_mae_gain_e75']:.6f} at E75 W6. Even after mapping, auxiliary MAE remains worse than R06 by {-mechanism_assessment['direct_real']['rank50_auxiliary_gain_vs_r06_e67']:.6f} and {-mechanism_assessment['direct_real']['rank50_auxiliary_gain_vs_r06_e75']:.6f}. E75's saved final policy harms R06 for all three arms.",
        f"REAL beats BAG/SHUFFLED by {mechanism_assessment['direct_real']['real_gain_vs_bag_e67']:.6f}/{mechanism_assessment['direct_real']['real_gain_vs_shuffled_e67']:.6f} at E67 W6, then trails them by {abs(mechanism_assessment['direct_real']['real_gain_vs_bag_e75']):.6f}/{abs(mechanism_assessment['direct_real']['real_gain_vs_shuffled_e75']):.6f} at E75 W6. That sign change is descriptive; every E75 direct arm still harms R06, and this audit cannot attribute the change causally to position.",
        "",
        f"The earlier residual REAL head has within-origin gain versus zero-residual of {mechanism_assessment['prior_residual_real']['raw_aux_gain_vs_zero_residual_e67']:.6f} at E67 and {mechanism_assessment['prior_residual_real']['raw_aux_gain_vs_zero_residual_e75']:.6f} at E75, with final policy gains {mechanism_assessment['prior_residual_real']['final_policy_gain_vs_r06_e67']:.6f} and {mechanism_assessment['prior_residual_real']['final_policy_gain_vs_r06_e75']:.6f}. Residual heads target different origin-specific residuals; their absolute losses are not compared across origins or to direct MAE.",
        "",
        "## Fixed groups",
        "",
        "`fixed_group_metrics.csv` uses disjoint fixed target groups (<=5, >5 and <95, >=95) and the already-frozen V4 fit-prefix text_load quartiles. Cells with n<100 are UNKNOWN; no bins were tuned or refit.",
        "",
        "## Historical novelty and next lane",
        "",
        evidence["historical_novelty"]["novelty_conclusion"],
        "",
        evidence["historical_novelty"]["reason"],
        "",
        "**Recommendation: NONE_JUSTIFIED.** Keep the frozen candidate FAIL. Any new model fit, preprocessing fit, rank/weight variation, E79 run, or submission would need separate authorization.",
        "",
        "## Scope and reproducibility",
        "",
        "No fitted model was loaded. No fit/predict call, threshold/weight grid, Test/Sample/Kaggle access, submission CSV/export, submission, ledger mutation, or source/cache write occurred. Six diagnostic CSV tables were written in this report directory; zero plots were created.",
        "",
        "Reproduce with the three explicit paths recorded in AUDIT.json and run `python scripts/reg_position_failure_mechanism_v1.py --source-root ... --saved-artifact-root ... --repo-root ...`. The output hashes and source signatures are in INPUT_HASHES.json and AUDIT.json.",
        "",
        "Input files and saved prediction/receipt bundles are SHA-256 pinned in INPUT_HASHES.json. Output CSV hashes are pinned in AUDIT.json.",
        "",
    ]
    (output_root / "MECHANISM_AUDIT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--saved-artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / OUTPUT_REL)
    args = parser.parse_args(argv)
    try:
        result = run(args.repo_root, args.source_root, args.saved_artifact_root, args.output_root)
        print(json.dumps({"status": result["status"], "audit_id": result["audit_id"],
                          "target_rows": result["source_boundary"]["query_target_rows_parsed"],
                          "score_blocks_reverified": len(result["comparison_checks"]),
                          "next_lane": result["historical_novelty"]["next_lane"]}, sort_keys=True))
        return 0
    except (AuditError, OSError, ValueError, KeyError) as exc:
        print(f"NO-GO: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
