"""Independent formula, lineage, persistence, and post-fit metric verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG_TRAIN = ROOT / "data/raw/regression/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
PREP = ROOT / "reports/optimization_v3/rnbr_pilot_v1/PREPARATION_PREFLIGHT.json"
FREEZE = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
BUNDLE_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1/bundles"
REPORT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json"
TRAIN_SHA = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
PILOT_ORIGINS = (67, 75)
PILOT_ARMS = ("observed", "shuffled")
FIT_IDS = tuple(f"RNBR_PILOT_E{o}_{a.upper()}" for o in PILOT_ORIGINS for a in PILOT_ARMS)
R06_PATHS = {
    55: "experiments/runs/optimization_v3_batch1/R06/R06_fit55.npz",
    59: "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit59.npz",
    63: "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit63.npz",
    67: "experiments/runs/optimization_v3_batch1/R06/R06_fit67.npz",
    71: "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit71.npz",
    75: "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit75.npz",
}
R22_W4 = "experiments/runs/optimization_v3_batch1/R22/R22_fit67"
R22_W6 = "experiments/runs/optimization_v3_final_continuation/R22/R22_fit75"
R06_VERIFICATION = "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json"


class VerificationError(RuntimeError):
    """An independent output/source boundary check failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha_array(value: Any, dtype: str | None = None) -> str:
    arr = np.asarray(value, dtype=dtype)
    if arr.dtype.kind in "OUS":
        h = hashlib.sha256()
        for item in arr.reshape(-1):
            body = str(item).encode("utf-8", errors="surrogatepass")
            h.update(len(body).to_bytes(8, "little"))
            h.update(body)
        return h.hexdigest()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


def _load_freeze() -> dict[str, Any]:
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    signature = frozen.get("freeze_signature")
    body = {k: v for k, v in frozen.items() if k != "freeze_signature"}
    if signature != hashlib.sha256(_canonical(body)).hexdigest():
        raise VerificationError("Freeze signature mismatch")
    for rel, expected in frozen.get("code_sha256", {}).items():
        if _sha_file(ROOT / rel) != expected:
            raise VerificationError(f"Frozen implementation source changed: {rel}")
    if _sha_file(REG_TRAIN) != TRAIN_SHA:
        raise VerificationError("Official Regression Train hash mismatch")
    prep = frozen["preparation_report"]
    protocol = frozen["protocol"]
    if (_sha_file(ROOT / prep["path"]) != prep["sha256"]
            or _sha_file(ROOT / protocol["path"]) != protocol["sha256"]
            or _sha_file(ROOT / frozen["r06_lineage"]["source_path"]) != frozen["r06_lineage"]["source_sha256"]
            or _sha_file(ROOT / frozen["r06_lineage"]["recipe_path"]) != frozen["r06_lineage"]["recipe_sha256"]):
        raise VerificationError("Frozen preparation/protocol/R06 source lineage changed")
    return frozen


def _train_metadata() -> dict[str, np.ndarray]:
    frame = pd.read_csv(REG_TRAIN, usecols=["record_id", "cycle_code", "signal_stream"])
    if len(frame) != 50608:
        raise VerificationError("Official Regression Train row count mismatch")
    ids = frame.record_id.astype(str).to_numpy()
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    present = frame.signal_stream.fillna("").astype(str).str.strip().ne("").to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise VerificationError("Official Train IDs are not unique")
    return {"ids": ids, "cycles": cycles, "present": present}


def _read_targets(rows: np.ndarray, *, purpose: str) -> dict[str, np.ndarray]:
    selected = np.asarray(rows, dtype=np.int64)
    if (selected.ndim != 1 or len(np.unique(selected)) != len(selected)
            or len(selected) == 0 or np.any(selected < 0) or np.any(selected >= 50608)):
        raise VerificationError(f"Invalid target allowlist for {purpose}")
    allowed = set(map(int, selected))
    frame = pd.read_csv(
        REG_TRAIN, usecols=["record_id", "cycle_code", "threat_rank"],
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in allowed,
    )
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    if (len(frame) != len(selected) or not np.isfinite(target).all()
            or np.any((target < 0) | (target > 100))):
        raise VerificationError(f"Target rows invalid for {purpose}")
    return {"ids": frame.record_id.astype(str).to_numpy(),
            "cycles": pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64),
            "target": target}


def _r06_cutoff(cycle: int) -> int:
    if 56 <= cycle <= 59:
        return 55
    if 60 <= cycle <= 63:
        return 59
    if 64 <= cycle <= 67:
        return 63
    if 68 <= cycle <= 71:
        return 67
    if 72 <= cycle <= 75:
        return 71
    if 76 <= cycle <= 79:
        return 75
    raise VerificationError(f"No strict-past R06 source cutoff for cycle {cycle}")


def _load_r06(train: dict[str, np.ndarray], freeze: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    output = {}
    for cutoff, rel in R06_PATHS.items():
        npz = ROOT / rel
        sidecar = npz.with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = freeze["r06_lineage"]["cache_inventory"][str(cutoff)]
        if (meta.get("npz_sha256") != _sha_file(npz)
                or meta.get("npz_sha256") != expected["npz_sha256"]
                or _sha_file(sidecar) != expected["sidecar_sha256"]):
            raise VerificationError(f"R06 cache hash mismatch at cutoff {cutoff}")
        identity = meta.get("identity", {})
        if (int(identity.get("cutoff", -1)) != cutoff
                or identity.get("card_id") != "R06"
                or (identity.get("freeze_signature") != expected["freeze_signature"]
                    and identity.get("campaign_freeze_signature") != expected["freeze_signature"])):
            raise VerificationError(f"R06 freeze/recipe identity mismatch at cutoff {cutoff}")
        with np.load(npz, allow_pickle=False) as z:
            rows = np.asarray(z["score_rows"], dtype=np.int64).copy()
            ids = np.asarray(z["score_record_ids"]).astype(str, copy=True)
            pred = np.asarray(z["prediction"], dtype=np.float64).copy()
            legacy = np.asarray(z["baseline_prediction"], dtype=np.float64).copy()
        if (not np.array_equal(ids, train["ids"][rows]) or len(rows) != len(pred)
                or len(rows) != len(legacy) or not np.isfinite(pred).all()
                or not np.isfinite(legacy).all()):
            raise VerificationError(f"R06 rows/IDs/predictions invalid at cutoff {cutoff}")
        output[cutoff] = {"rows": rows, "ids": ids, "prediction": pred,
                          "legacy": legacy, "cycles": train["cycles"][rows]}
    return output


def _r06_for_rows(rows: np.ndarray, cycles: np.ndarray, cache: dict[int, dict[str, np.ndarray]],
                  field: str) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float64)
    source_cutoffs = np.asarray([_r06_cutoff(int(c)) for c in cycles], dtype=np.int64)
    for cutoff in np.unique(source_cutoffs):
        positions = np.flatnonzero(source_cutoffs == cutoff)
        mapping = {int(r): float(v) for r, v in zip(cache[int(cutoff)]["rows"], cache[int(cutoff)][field])}
        try:
            out[positions] = [mapping[int(rows[p])] for p in positions]
        except KeyError as exc:
            raise VerificationError(f"Missing R06 row at cutoff {cutoff}: {exc.args[0]}") from exc
    return out


def _geometry(origin: int, freeze: dict[str, Any], train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arms = freeze["a02_saved_geometry"][str(origin)]
    names = ("score_rows", "score_record_ids", "score_cycles", "neighbor_count",
             "neighbor_source_rows", "neighbor_ids", "neighbor_similarities")
    arm_values: dict[str, dict[str, np.ndarray]] = {}
    for arm in PILOT_ARMS:
        spec = arms[arm]
        path = ROOT / spec["npz"]
        if (_sha_file(path) != spec["npz_sha256"]
                or _sha_file(ROOT / spec["model"]) != spec["model_sha256"]
                or _sha_file(ROOT / spec["predictions_json"]) != spec["predictions_json_sha256"]
                or _sha_file(ROOT / spec["receipt_json"]) != spec["receipt_json_sha256"]
                or _sha_file(ROOT / spec["state_json"]) != spec["state_json_sha256"]):
            raise VerificationError(f"A02 frozen artifact hash mismatch E{origin}/{arm}")
        with np.load(path, allow_pickle=False) as z:
            arm_values[arm] = {name: np.asarray(z[name]).copy() for name in names}
    for name in names:
        if not np.array_equal(arm_values["observed"][name], arm_values["shuffled"][name]):
            raise VerificationError(f"Observed/shuffled A02 geometries differ E{origin}/{name}")
    values = arm_values["observed"]
    for arm in PILOT_ARMS:
        spec = arms[arm]
        pred_meta = json.loads((ROOT / spec["predictions_json"]).read_text(encoding="utf-8"))
        receipt = json.loads((ROOT / spec["receipt_json"]).read_text(encoding="utf-8"))
        state = json.loads((ROOT / spec["state_json"]).read_text(encoding="utf-8"))
        receipt_body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        state_body = {k: v for k, v in state.items() if k != "state_sha256"}
        if (pred_meta.get("npz_sha256") != spec["npz_sha256"]
                or pred_meta.get("fit_id") != spec["fit_id"]
                or pred_meta.get("residual_arm") != arm
                or pred_meta.get("freeze_signature") != spec["freeze_signature"]
                or receipt.get("receipt_sha256") != hashlib.sha256(_canonical(receipt_body)).hexdigest()
                or state.get("state_sha256") != hashlib.sha256(_canonical(state_body)).hexdigest()
                or receipt.get("query_labels_passed_to_fit") is not False
                or receipt.get("test_data_used") is not False):
            raise VerificationError(f"A02 target-free geometry lineage failed E{origin}/{arm}")
    rows = values["score_rows"].astype(np.int64)
    if (not np.array_equal(values["score_record_ids"].astype(str), train["ids"][rows])
            or not np.array_equal(values["score_cycles"].astype(np.int64), train["cycles"][rows])):
        raise VerificationError(f"A02 geometry IDs/cycles mismatch at E{origin}")
    return values


def _independent_shuffle(residual: np.ndarray, cycles: np.ndarray, prediction: np.ndarray,
                         source_rows: np.ndarray) -> np.ndarray:
    out = np.asarray(residual, dtype=np.float64).copy()
    deciles = np.minimum(9, np.floor(np.clip(prediction, 0.0, 100.0) / 10.0)).astype(np.int64)
    rng = np.random.default_rng(42)
    for cycle in sorted(set(map(int, cycles))):
        for decile in range(10):
            members = [i for i in range(len(out)) if int(cycles[i]) == cycle and int(deciles[i]) == decile]
            members.sort(key=lambda i: int(source_rows[i]))
            if len(members) > 1:
                permutation = rng.permutation(len(members))
                original = np.asarray([residual[i] for i in members], dtype=np.float64)
                for target_position, source_position in enumerate(permutation):
                    out[members[target_position]] = original[int(source_position)]
    return out


def _weighted_median_independent(residual: np.ndarray, cosine: np.ndarray, rows: np.ndarray) -> float:
    records = sorted(((float(residual[i]), int(rows[i]), float(cosine[i]) ** 2)
                       for i in range(len(residual))), key=lambda item: (item[0], item[1]))
    grouped: list[tuple[float, float]] = []
    for value, _row, weight in records:
        if grouped and value == grouped[-1][0]:
            grouped[-1] = (value, grouped[-1][1] + weight)
        else:
            grouped.append((value, weight))
    total = math.fsum(weight for _, weight in grouped)
    if not math.isfinite(total) or total <= 0:
        raise VerificationError("Invalid neighbor weight sum")
    half, tolerance, cumulative = total / 2.0, total * 1e-12, 0.0
    for index, (value, weight) in enumerate(grouped):
        cumulative += weight
        if cumulative > half + tolerance:
            return value
        if abs(cumulative - half) <= tolerance:
            return (value + grouped[index + 1][0]) / 2.0 if index + 1 < len(grouped) else value
    return grouped[-1][0]


def independent_prediction_recompute(arrays: dict[str, np.ndarray],
                                     donor_residuals_used: np.ndarray | None = None
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild deltas/predictions without importing the production predictor."""
    donor_rows = np.asarray(arrays["donor_rows"], dtype=np.int64)
    residual = np.asarray(arrays["donor_residuals_used"] if donor_residuals_used is None
                          else donor_residuals_used, dtype=np.float64)
    baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    missing = np.asarray(arrays["query_missing"], dtype=bool)
    counts = np.asarray(arrays["neighbor_count"], dtype=np.int64)
    neighbor_rows = np.asarray(arrays["neighbor_source_rows"], dtype=np.int64)
    similarities = np.asarray(arrays["neighbor_similarities"], dtype=np.float64)
    if (residual.shape != donor_rows.shape or baseline.shape != missing.shape
            or counts.shape != baseline.shape or neighbor_rows.shape != similarities.shape
            or neighbor_rows.shape[0] != len(baseline)):
        raise VerificationError("Independent formula arrays are misaligned")
    lookup = {int(row): index for index, row in enumerate(donor_rows)}
    delta = np.zeros(len(baseline), dtype=np.float64)
    for i, count in enumerate(counts):
        k = int(count)
        if missing[i] or k < 5:
            continue
        rows = neighbor_rows[i, :k].astype(np.int64)
        positions = np.asarray([lookup[int(row)] for row in rows], dtype=np.int64)
        median = _weighted_median_independent(residual[positions], similarities[i, :k], rows)
        delta[i] = float(np.clip(0.25 * median, -5.0, 5.0))
    return delta, np.clip(baseline + delta, 0.0, 100.0)


def _load_bundle(fit_id: str, freeze: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folder = BUNDLE_ROOT / fit_id
    npz, receipt_path = folder / "arrays.npz", folder / "receipt.json"
    if not npz.is_file() or not receipt_path.is_file():
        raise VerificationError(f"Missing persisted bundle: {fit_id}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = receipt.get("identity", {})
    body = {k: v for k, v in receipt.items() if k != "integrity_sha256"}
    if (receipt.get("integrity_sha256") != hashlib.sha256(_canonical(body)).hexdigest()
            or receipt.get("npz_sha256") != _sha_file(npz)
            or identity.get("fit_id") != fit_id
            or identity.get("freeze_signature") != freeze["freeze_signature"]):
        raise VerificationError(f"Bundle receipt/hash mismatch: {fit_id}")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]).copy() for key in z.files}
    return arrays, receipt


def verify_fit(fit_id: str, freeze: dict[str, Any], train: dict[str, np.ndarray],
               cache: dict[int, dict[str, np.ndarray]]) -> dict[str, Any]:
    identity_bits = fit_id.split("_")
    origin = int(identity_bits[2][1:])
    arm = identity_bits[3].lower()
    arrays, receipt = _load_bundle(fit_id, freeze)
    donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin)
                                & train["present"]).astype(np.int64)
    donor_cycles = train["cycles"][donor_rows]
    donor_ids = train["ids"][donor_rows]
    donor_values = _read_targets(donor_rows, purpose=f"{fit_id}_DONOR_RECOMPUTATION")
    if (not np.array_equal(donor_values["ids"], donor_ids)
            or not np.array_equal(donor_values["cycles"], donor_cycles)):
        raise VerificationError(f"Donor labels do not align: {fit_id}")
    donor_r06 = _r06_for_rows(donor_rows, donor_cycles, cache, "prediction")
    expected_residuals = donor_values["target"] - donor_r06
    expected_used = (expected_residuals if arm == "observed" else
                     _independent_shuffle(expected_residuals, donor_cycles, donor_r06, donor_rows))
    if (not np.array_equal(arrays["donor_rows"], donor_rows)
            or not np.array_equal(arrays["donor_record_ids"].astype(str), donor_ids)
            or not np.array_equal(arrays["donor_cycles"], donor_cycles)
            or not np.array_equal(arrays["donor_source_cutoffs"], np.asarray([_r06_cutoff(int(c)) for c in donor_cycles]))
            or not np.array_equal(arrays["donor_r06_predictions"], donor_r06)
            or not np.array_equal(arrays["donor_residuals"], expected_residuals)
            or not np.array_equal(arrays["donor_residuals_used"], expected_used)):
        raise VerificationError(f"Donor value lineage/residual vector mismatch: {fit_id}")
    geometry = _geometry(origin, freeze, train)
    score_rows = geometry["score_rows"].astype(np.int64)
    score_cycles = train["cycles"][score_rows]
    score_ids = train["ids"][score_rows]
    missing = ~train["present"][score_rows]
    baseline = _r06_for_rows(score_rows, score_cycles, cache, "prediction")
    counts = geometry["neighbor_count"].astype(np.int64)
    expected_delta, expected_prediction = independent_prediction_recompute(
        arrays, donor_residuals_used=expected_used)
    if (not np.array_equal(arrays["score_rows"], score_rows)
            or not np.array_equal(arrays["score_record_ids"].astype(str), score_ids)
            or not np.array_equal(arrays["score_cycles"], score_cycles)
            or not np.array_equal(arrays["baseline_prediction"], baseline)
            or not np.array_equal(arrays["query_missing"], missing)
            or not np.array_equal(arrays["neighbor_count"], counts)
            or not np.array_equal(arrays["neighbor_source_rows"], geometry["neighbor_source_rows"])
            or not np.array_equal(arrays["neighbor_ids"].astype(str), geometry["neighbor_ids"].astype(str))
            or not np.array_equal(arrays["neighbor_similarities"], geometry["neighbor_similarities"])
            or not np.allclose(arrays["delta"], expected_delta, atol=1e-12, rtol=0)
            or not np.allclose(arrays["prediction"], expected_prediction, atol=1e-12, rtol=0)
            or not np.array_equal(arrays["prediction"][missing], baseline[missing])):
        raise VerificationError(f"Independent saved-neighbor prediction reconstruction failed: {fit_id}")
    return {"fit_id": fit_id, "origin": origin, "arm": arm, "status": "PASS",
            "donor_rows": int(len(donor_rows)), "query_rows": int(len(score_rows)),
            "neighbor_rows_checked": int(np.sum(counts)),
            "missing_queries_unchanged": int(np.sum(missing)),
            "bundle_npz_sha256": receipt["npz_sha256"],
            "receipt_sha256": receipt["integrity_sha256"],
            "donor_residual_sha256_recomputed": _sha_array(expected_residuals, "<f8"),
            "donor_target_read_scope": "present-stream Regression threat_rank donor rows only"}


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) != len(prediction) or not len(y) or not np.isfinite(prediction).all():
        raise VerificationError("Metric input arrays are invalid")
    return float(np.mean(np.abs(y - prediction)))


def _round_metric(value: float) -> float:
    return float(round(value, 12))


def _aligned_cache(path_stem: str, prediction_key: str, train: dict[str, np.ndarray],
                   expected_rows: np.ndarray, *, expected_card: str,
                   expected_cutoff: int, expected_npz_sha256: str,
                   expected_sidecar_sha256: str) -> tuple[np.ndarray, dict[str, Any]]:
    npz, sidecar_path = ROOT / f"{path_stem}.npz", ROOT / f"{path_stem}.json"
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    identity = meta.get("identity", {})
    integrity = {k: v for k, v in meta.items() if k != "integrity_sha256"}
    if (meta.get("npz_sha256") != _sha_file(npz)
            or _sha_file(npz) != expected_npz_sha256
            or _sha_file(sidecar_path) != expected_sidecar_sha256
            or meta.get("identity_sha256") != hashlib.sha256(_canonical(identity)).hexdigest()
            or meta.get("integrity_sha256") != hashlib.sha256(_canonical(integrity)).hexdigest()
            or identity.get("card_id") != expected_card
            or int(identity.get("cutoff", -1)) != expected_cutoff):
        raise VerificationError(f"Comparator cache hash mismatch: {path_stem}")
    with np.load(npz, allow_pickle=False) as z:
        rows = np.asarray(z["score_rows"], dtype=np.int64).copy()
        ids = np.asarray(z["score_record_ids"]).astype(str, copy=True)
        pred = np.asarray(z[prediction_key], dtype=np.float64).copy()
    if (not np.array_equal(ids, train["ids"][rows]) or not np.isfinite(pred).all()
            or not np.array_equal(rows, expected_rows)):
        raise VerificationError(f"Comparator row/ID/finite alignment failed: {path_stem}/{prediction_key}")
    return pred, {"path": path_stem + ".npz", "npz_sha256": _sha_file(npz),
                  "sidecar_sha256": _sha_file(sidecar_path),
                  "fit_identity": {k: meta.get("identity", {}).get(k)
                                   for k in ("card_id", "cutoff", "experiment_id", "freeze_signature")},
                  "prediction_key": prediction_key}


def _compare_predictions(freeze: dict[str, Any], train: dict[str, np.ndarray],
                         fit_results: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, Any], dict[str, Any]]:
    e67 = fit_results["RNBR_PILOT_E67_OBSERVED"]
    e67_shuffle = fit_results["RNBR_PILOT_E67_SHUFFLED"]
    e75 = fit_results["RNBR_PILOT_E75_OBSERVED"]
    e75_shuffle = fit_results["RNBR_PILOT_E75_SHUFFLED"]
    e67_rows = e67["score_rows"].astype(np.int64)
    e75_rows = e75["score_rows"].astype(np.int64)
    e67_cycles = e67["score_cycles"].astype(np.int64)
    e75_cycles = e75["score_cycles"].astype(np.int64)
    w4_mask = (e67_cycles >= 68) & (e67_cycles <= 71)
    w6_mask = (e75_cycles >= 76) & (e75_cycles <= 79)
    primary_rows = np.concatenate((e67_rows[w4_mask], e75_rows[w6_mask]))
    h12_rows = e67_rows.copy()
    h12_y = _read_targets(h12_rows, purpose="E67_H12_L_QUERY_EVALUATION")
    if not np.array_equal(h12_y["ids"], train["ids"][h12_rows]):
        raise VerificationError("Query truth IDs are not aligned to frozen prediction rows")
    h12_positions = {int(row): pos for pos, row in enumerate(h12_rows)}
    y_primary = np.asarray([h12_y["target"][h12_positions[int(row)]] for row in primary_rows], dtype=np.float64)
    y_h12 = h12_y["target"]
    candidate_primary = np.concatenate((e67["prediction"][w4_mask], e75["prediction"][w6_mask]))
    shuffle_primary = np.concatenate((e67_shuffle["prediction"][w4_mask],
                                       e75_shuffle["prediction"][w6_mask]))
    r06_primary = np.concatenate((e67["baseline_prediction"][w4_mask],
                                  e75["baseline_prediction"][w6_mask]))
    r22_w4, r22w4_id = _aligned_cache(R22_W4, "prediction", train, e67_rows,
                                      expected_card="R22", expected_cutoff=67,
                                      expected_npz_sha256=freeze["evaluation_references"]["R22_W4"]["npz_sha256"],
                                      expected_sidecar_sha256=freeze["evaluation_references"]["R22_W4"]["sidecar_sha256"])
    r22_w6, r22w6_id = _aligned_cache(R22_W6, "R22_ORIGINAL", train, e75_rows,
                                      expected_card="R22", expected_cutoff=75,
                                      expected_npz_sha256=freeze["evaluation_references"]["R22_W6"]["npz_sha256"],
                                      expected_sidecar_sha256=freeze["evaluation_references"]["R22_W6"]["sidecar_sha256"])
    r22_primary = np.concatenate((r22_w4[w4_mask], r22_w6[w6_mask]))
    r22_h12 = r22_w4
    legacy_w4, legacy_w4_id = _aligned_cache(
        "experiments/runs/optimization_v3_batch1/R06/R06_fit67", "baseline_prediction", train, e67_rows,
        expected_card="R06", expected_cutoff=67,
        expected_npz_sha256=freeze["r06_lineage"]["cache_inventory"]["67"]["npz_sha256"],
        expected_sidecar_sha256=freeze["r06_lineage"]["cache_inventory"]["67"]["sidecar_sha256"])
    legacy_w6, legacy_w6_id = _aligned_cache(
        "experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit75",
        "baseline_prediction", train, e75_rows, expected_card="R06", expected_cutoff=75,
        expected_npz_sha256=freeze["r06_lineage"]["cache_inventory"]["75"]["npz_sha256"],
        expected_sidecar_sha256=freeze["r06_lineage"]["cache_inventory"]["75"]["sidecar_sha256"])
    legacy_primary = np.concatenate((legacy_w4[w4_mask], legacy_w6[w6_mask]))
    legacy_h12 = legacy_w4
    r06_h12 = e67["baseline_prediction"]
    models_primary = {"R06": r06_primary, "RNBR_OBSERVED": candidate_primary,
                      "RNBR_SHUFFLED": shuffle_primary, "R22_ORIGINAL_CACHED": r22_primary,
                      "R06_LEGACY_BASELINE": legacy_primary}
    models_h12 = {"R06": r06_h12, "RNBR_OBSERVED": e67["prediction"],
                  "RNBR_SHUFFLED": e67_shuffle["prediction"],
                  "R22_ORIGINAL_CACHED": r22_h12, "R06_LEGACY_BASELINE": legacy_h12}
    primary_models = {name: _round_metric(_mae(y_primary, p)) for name, p in models_primary.items()}
    h12_models = {name: _round_metric(_mae(y_h12, p)) for name, p in models_h12.items()}
    def view(rows: np.ndarray, truth: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"n": int(len(rows)), "cycle_range": [int(train["cycles"][rows].min()),
                                                      int(train["cycles"][rows].max())],
                "mae": {name: _round_metric(_mae(truth, value)) for name, value in preds.items()}}
    views = {
        "primary_W4_plus_W6": {"n": int(len(primary_rows)), "mae": primary_models},
        "H12_L_E67_only": {"n": int(len(h12_rows)), "cycle_range": [68, 79], "mae": h12_models},
        "windows": {
            "W4_E67_68_71": view(e67_rows[w4_mask], h12_y["target"][np.searchsorted(h12_rows, e67_rows[w4_mask])],
                                   {"R06": e67["baseline_prediction"][w4_mask],
                                    "RNBR_OBSERVED": e67["prediction"][w4_mask],
                                    "RNBR_SHUFFLED": e67_shuffle["prediction"][w4_mask],
                                    "R22_ORIGINAL_CACHED": r22_w4[w4_mask],
                                    "R06_LEGACY_BASELINE": legacy_w4[w4_mask]}),
            "W5_E67_72_75": view(e67_rows[(e67_cycles >= 72) & (e67_cycles <= 75)],
                                   h12_y["target"][np.searchsorted(h12_rows, e67_rows[(e67_cycles >= 72) & (e67_cycles <= 75)])],
                                   {"R06": e67["baseline_prediction"][(e67_cycles >= 72) & (e67_cycles <= 75)],
                                    "RNBR_OBSERVED": e67["prediction"][(e67_cycles >= 72) & (e67_cycles <= 75)],
                                    "RNBR_SHUFFLED": e67_shuffle["prediction"][(e67_cycles >= 72) & (e67_cycles <= 75)],
                                    "R22_ORIGINAL_CACHED": r22_w4[(e67_cycles >= 72) & (e67_cycles <= 75)],
                                    "R06_LEGACY_BASELINE": legacy_w4[(e67_cycles >= 72) & (e67_cycles <= 75)]}),
            "W6_E67_76_79": view(e67_rows[(e67_cycles >= 76) & (e67_cycles <= 79)],
                                   h12_y["target"][np.searchsorted(h12_rows, e67_rows[(e67_cycles >= 76) & (e67_cycles <= 79)])],
                                   {"R06": e67["baseline_prediction"][(e67_cycles >= 76) & (e67_cycles <= 79)],
                                    "RNBR_OBSERVED": e67["prediction"][(e67_cycles >= 76) & (e67_cycles <= 79)],
                                    "RNBR_SHUFFLED": e67_shuffle["prediction"][(e67_cycles >= 76) & (e67_cycles <= 79)],
                                    "R22_ORIGINAL_CACHED": r22_w4[(e67_cycles >= 76) & (e67_cycles <= 79)],
                                    "R06_LEGACY_BASELINE": legacy_w4[(e67_cycles >= 76) & (e67_cycles <= 79)]}),
            "W6_E75_76_79": view(e75_rows[w6_mask], y_primary[len(e67_rows[w4_mask]):],
                                   {"R06": e75["baseline_prediction"][w6_mask],
                                    "RNBR_OBSERVED": e75["prediction"][w6_mask],
                                    "RNBR_SHUFFLED": e75_shuffle["prediction"][w6_mask],
                                    "R22_ORIGINAL_CACHED": r22_w6[w6_mask],
                                    "R06_LEGACY_BASELINE": legacy_w6[w6_mask]}),
        },
        "per_cycle_H12_L_E67": {},
        "subgroups_primary": {},
        "subgroups_H12_L": {},
        "gains_and_harms": {},
    }
    for cycle in range(68, 80):
        mask = e67_cycles == cycle
        y = h12_y["target"][mask]
        views["per_cycle_H12_L_E67"][str(cycle)] = {
            "n": int(mask.sum()), "R06_mae": _round_metric(_mae(y, e67["baseline_prediction"][mask])),
            "observed_mae": _round_metric(_mae(y, e67["prediction"][mask])),
            "shuffled_mae": _round_metric(_mae(y, e67_shuffle["prediction"][mask])),
            "observed_gain_vs_R06": _round_metric(_mae(y, e67["baseline_prediction"][mask]) - _mae(y, e67["prediction"][mask])),
        }
    primary_missing = np.concatenate((e67["query_missing"][w4_mask], e75["query_missing"][w6_mask]))
    groups = {"missing_stream": primary_missing,
              "threat_rank_le_5": y_primary <= 5.0,
              "threat_rank_ge_95": y_primary >= 95.0}
    for name, mask in groups.items():
        mask = np.asarray(mask, dtype=bool)
        views["subgroups_primary"][name] = {
            "n": int(mask.sum()), "supported_n_ge_100": bool(mask.sum() >= 100),
            "mae": {model: (_round_metric(_mae(y_primary[mask], values[mask])) if mask.any() else None)
                    for model, values in models_primary.items()},
        }
    h12_groups = {"missing_stream": e67["query_missing"],
                  "threat_rank_le_5": y_h12 <= 5.0,
                  "threat_rank_ge_95": y_h12 >= 95.0}
    for name, mask in h12_groups.items():
        mask = np.asarray(mask, dtype=bool)
        views["subgroups_H12_L"][name] = {
            "n": int(mask.sum()), "supported_n_ge_100": bool(mask.sum() >= 100),
            "mae": {model: (_round_metric(_mae(y_h12[mask], values[mask])) if mask.any() else None)
                    for model, values in models_h12.items()},
        }
    gains = {
        "primary_gain_vs_R06": primary_models["R06"] - primary_models["RNBR_OBSERVED"],
        "primary_gain_vs_shuffled": primary_models["RNBR_SHUFFLED"] - primary_models["RNBR_OBSERVED"],
        "H12_L_gain_vs_R06": h12_models["R06"] - h12_models["RNBR_OBSERVED"],
        "H12_L_gain_vs_shuffled": h12_models["RNBR_SHUFFLED"] - h12_models["RNBR_OBSERVED"],
        "W4_harm_vs_R06": views["windows"]["W4_E67_68_71"]["mae"]["RNBR_OBSERVED"] - views["windows"]["W4_E67_68_71"]["mae"]["R06"],
        "W6_E75_harm_vs_R06": views["windows"]["W6_E75_76_79"]["mae"]["RNBR_OBSERVED"] - views["windows"]["W6_E75_76_79"]["mae"]["R06"],
    }
    checks = {
        "primary_gain_vs_R06_at_least_0_03": gains["primary_gain_vs_R06"] >= 0.03,
        "primary_gain_vs_shuffle_at_least_0_01": gains["primary_gain_vs_shuffled"] >= 0.01,
        "H12_L_harm_vs_R06_at_most_0_05": -gains["H12_L_gain_vs_R06"] <= 0.05,
        "W4_harm_vs_R06_below_0_25": gains["W4_harm_vs_R06"] < 0.25,
        "W6_harm_vs_R06_below_0_25": gains["W6_E75_harm_vs_R06"] < 0.25,
    }
    for name in ("missing_stream", "threat_rank_le_5", "threat_rank_ge_95"):
        group = views["subgroups_primary"][name]
        if group["supported_n_ge_100"]:
            harm = group["mae"]["RNBR_OBSERVED"] - group["mae"]["R06"]
            gains[f"{name}_harm_vs_R06"] = harm
            checks[f"{name}_harm_vs_R06_at_most_0_25"] = harm <= 0.25
    missing_group = views["subgroups_primary"]["missing_stream"]
    if missing_group["n"]:
        checks["missing_stream_incremental_harm_exactly_zero"] = (
            missing_group["mae"]["RNBR_OBSERVED"] == missing_group["mae"]["R06"])
    views["gains_and_harms"] = {key: _round_metric(value) for key, value in gains.items()}
    views["diagnostic_gates"] = {"checks": checks, "all_pass": bool(all(checks.values())),
                                 "qualified_model_claim": False,
                                 "interpretation": "R-NBR pilot is diagnostic-only regardless of gates."}
    historical_refs = freeze["evaluation_references"]
    if (_sha_file(ROOT / historical_refs["R06_historical_results"]["path"])
            != historical_refs["R06_historical_results"]["sha256"]
            or _sha_file(ROOT / historical_refs["R06_historical_verification"]["path"])
            != historical_refs["R06_historical_verification"]["sha256"]):
        raise VerificationError("Frozen R06 historical guard reports changed")
    r06_historical = json.loads((ROOT / R06_VERIFICATION).read_text(encoding="utf-8"))
    r06_results_path = ROOT / "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json"
    r06_results = json.loads(r06_results_path.read_text(encoding="utf-8"))
    archived_missing = r06_results["groups"]["views"]["W1"]["signal_stream_missing"]
    if (archived_missing["n"] != 392 or abs(archived_missing["harm"] - 0.3955562085105022) > 1e-12
            or archived_missing["pass"] is not False):
        raise VerificationError("Historical R06 W1 missing-stream guard reference changed")
    views["historical_R06_legacy_guard"] = {
        "source": R06_VERIFICATION, "source_sha256": _sha_file(ROOT / R06_VERIFICATION),
        "results_source": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/results.json",
        "results_source_sha256": _sha_file(r06_results_path),
        "R06_qualified": r06_historical.get("qualification", {}).get("qualified"),
        "archived_W1_missing_stream": archived_missing,
        "archived_supported_group_failures": r06_historical.get("qualification", {}).get("supported_group_failures"),
        "kept_separate_from_this_late_incremental_gate": True,
    }
    comparison_lineage = {"R22_W4": r22w4_id, "R22_W6": r22w6_id,
                          "R06_legacy_W4": legacy_w4_id, "R06_legacy_W6": legacy_w6_id,
                          "R22_W4_label": "saved R22 cutoff-67 prediction from batch1",
                          "R22_W6_label": "saved R22_ORIGINAL cutoff-75 prediction from final continuation",
                          "rows_aligned": True}
    return views, comparison_lineage


def verify(*, all_fits: bool) -> dict[str, Any]:
    freeze = _load_freeze()
    train = _train_metadata()
    cache = _load_r06(train, freeze)
    folders = {fit_id for fit_id in FIT_IDS if (BUNDLE_ROOT / fit_id).is_dir()}
    if all_fits and folders != set(FIT_IDS):
        raise VerificationError(f"Expected four fit bundles; found {sorted(folders)}")
    fit_report = [verify_fit(fit_id, freeze, train, cache) for fit_id in FIT_IDS if fit_id in folders]
    if len(fit_report) != len(folders):
        raise VerificationError("Not every saved fit received an independent formula audit")
    result: dict[str, Any] = {
        "status": "PASS", "freeze_signature": freeze["freeze_signature"],
        "fit_bundle_verification": fit_report,
        "query_truth_loaded_after_all_four_fit_bundles_verified": False,
        "classification_labels_loaded": False, "test_or_sample_submission_opened": False,
    }
    if all_fits:
        if set(item["fit_id"] for item in fit_report) != set(FIT_IDS):
            raise VerificationError("All four fit IDs were not verified before query truth access")
        # This is the first access to query target values. All four frozen outputs
        # and their independent formula/lineage checks have already passed above.
        fit_results = {fit_id: _load_bundle(fit_id, freeze)[0] for fit_id in FIT_IDS}
        metrics, comparator_lineage = _compare_predictions(freeze, train, fit_results)
        result["query_truth_loaded_after_all_four_fit_bundles_verified"] = True
        result["metrics_and_views"] = metrics
        result["comparison_lineage"] = comparator_lineage
        result["source_boundaries"] = {
            "official_train_only": True, "query_truth_rows": 20903,
            "query_truth_cycles": [68, 79], "query_truth_loaded_after_bundle_verification": True,
            "test_or_sample_submission_opened": False, "classification_labels_loaded": False,
            "kaggle_access": False, "full_train_refit": False,
        }
        # Report raw per-bundle integrity hashes and provenance without retaining labels.
        result["bundle_receipts"] = [{"fit_id": i, "npz_sha256": _load_bundle(i, freeze)[1]["npz_sha256"],
                                      "receipt_sha256": _load_bundle(i, freeze)[1]["integrity_sha256"]}
                                     for i in FIT_IDS]
        _write_once(REPORT, result)
    return result


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise VerificationError(f"Refusing to overwrite different independent verification: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify-pending", action="store_true")
    modes.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()
    result = verify(all_fits=args.verify_all)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
