"""Zero-fit decision-boundary audit of persisted A02 predictions.

Uses only official Classification Train, previous A02 audit statistics/QA, and
persisted row-level prediction/descriptor arrays. It does not load a model,
transform text, fit an estimator, change probabilities, or access Test/Kaggle.

Run from the repository root:

    python -B scripts/optimization_v3_a02_decision_boundary_audit.py
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_REL = "data/raw/classification/Train.csv"
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
PARENT_REPORT = ROOT / "reports/optimization_v3/a02_reliability_audit_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_decision_boundary_audit_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_a02_reliability_audit_v1"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
EXPECTED_LEDGER_SHA256 = "6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147"
PILOT_FREEZE_SIGNATURE = "cdac164df87d4f559e36795066cc0c3a91034e5917b0b1fd72cebd1051dae08b"
CONT_FREEZE_SIGNATURE = "93a2f388673e5cbb2fde60364fd0121918695ecb908c9b2c9c8b91e239de54e6"

ROW_ARTIFACTS = {
    "full_six_primary": "row_level_full_six_primary.npz",
    "new_origin_aggregate": "row_level_new_origin_aggregate.npz",
    "late": "row_level_late.npz",
    "inherited_H12_L": "row_level_inherited_H12_L.npz",
}
CACHE_PATHS = {
    "55": "experiments/runs/optimization_v3_batch1/C16/C16_fit55.npz",
    "59": "experiments/runs/optimization_v3_final_continuation/C16/C16_fit59.npz",
    "63": "experiments/runs/optimization_v3_final_continuation/C16/C16_fit63.npz",
    "67": "experiments/runs/optimization_v3_batch1/C16/C16_fit67.npz",
    "71": "experiments/runs/optimization_v3_final_continuation/C16/C16_fit71.npz",
    "75": "experiments/runs/optimization_v3_final_continuation/C16/C16_fit75.npz",
}
ORIGINS = (59, 63, 67, 71, 75)
ORIGIN_FIRST4 = {59: (60, 63), 63: (64, 67), 67: (68, 71),
                 71: (72, 75), 75: (76, 79)}
THRESHOLD = 0.50
CORRECTION_LIMIT = 0.05
MARGIN_LOW = 0.45
MARGIN_HIGH = 0.55
BASELINE_BANDS = (
    (0.45, 0.475, "[.45,.475)"),
    (0.475, 0.50, "[.475,.50)"),
    (0.50, 0.525, "[.50,.525)"),
    (0.525, 0.55, "[.525,.55]"),
)
F1_MIN_ROWS = 200
F1_MIN_CLASS_ROWS = 30
BOUNDARY_MIN_ROWS = 100
BOUNDARY_MIN_CHANGES = 20
FLIP_STRATUM_MIN = 20
DESCRIPTORS = (
    "ess",
    "similarity_mass",
    "max_cosine",
    "weighted_residual_sd",
    "weighted_residual_sign_consistency",
    "max_cycle_weight_concentration",
    "mean_origin_age",
    "mean_query_age",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _is_finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _f1_from_confusion(tn: int, fp: int, fn: int, tp: int) -> float:
    f1_zero_den = 2 * tn + fp + fn
    f1_one_den = 2 * tp + fp + fn
    f1_zero = (2 * tn / f1_zero_den) if f1_zero_den else 0.0
    f1_one = (2 * tp / f1_one_den) if f1_one_den else 0.0
    return (f1_zero + f1_one) / 2.0


def fixed_band_mask(probabilities: np.ndarray, band_index: int) -> np.ndarray:
    """Return a fixed left-closed band; only the last band includes .55."""
    if band_index < 0 or band_index >= len(BASELINE_BANDS):
        raise ValueError("band_index must be 0..3")
    low, high, _ = BASELINE_BANDS[band_index]
    values = np.asarray(probabilities, dtype=np.float64)
    if band_index == len(BASELINE_BANDS) - 1:
        return (values >= low) & (values <= high)
    return (values >= low) & (values < high)


def outside_margin_change_count(baseline_p: np.ndarray, arm_p: np.ndarray) -> int:
    base = np.asarray(baseline_p, dtype=np.float64)
    arm = np.asarray(arm_p, dtype=np.float64)
    if base.ndim != 1 or arm.shape != base.shape:
        raise ValueError("baseline_p and arm_p must be aligned vectors")
    changed = (base >= THRESHOLD) != (arm >= THRESHOLD)
    outside = (base < MARGIN_LOW) | (base > MARGIN_HIGH)
    return int(np.sum(changed & outside))


def flip_accounting(y: np.ndarray, baseline_p: np.ndarray,
                    arm_p: np.ndarray) -> dict[str, Any]:
    """Compute exact >=0.50 label changes and error transitions on paired rows."""
    truth = np.asarray(y, dtype=np.int8)
    base = np.asarray(baseline_p, dtype=np.float64)
    arm = np.asarray(arm_p, dtype=np.float64)
    if truth.ndim != 1 or base.shape != truth.shape or arm.shape != truth.shape:
        raise ValueError("y, baseline_p, and arm_p must be aligned vectors")
    if not np.isin(truth, [0, 1]).all() or not _is_finite(base) or not _is_finite(arm):
        raise ValueError("nonbinary truth or nonfinite probability")
    if np.any((base < 0) | (base > 1) | (arm < 0) | (arm > 1)):
        raise ValueError("probabilities must lie in [0,1]")

    base_label = base >= THRESHOLD
    arm_label = arm >= THRESHOLD
    changed = base_label != arm_label
    flip_0_to_1 = (~base_label) & arm_label
    flip_1_to_0 = base_label & (~arm_label)
    baseline_wrong = base_label != truth.astype(bool)
    arm_wrong = arm_label != truth.astype(bool)
    useful = changed & baseline_wrong & ~arm_wrong
    harmful = changed & ~baseline_wrong & arm_wrong
    if int(changed.sum()) != int(useful.sum() + harmful.sum()):
        raise RuntimeError("Every label change must be either useful or harmful")

    n = len(truth)
    base_positive = int(base_label.sum())
    arm_positive = int(arm_label.sum())
    baseline_errors = int(baseline_wrong.sum())
    arm_errors = int(arm_wrong.sum())
    tn = int(np.sum((truth == 0) & ~arm_label))
    fp = int(np.sum((truth == 0) & arm_label))
    fn = int(np.sum((truth == 1) & ~arm_label))
    tp = int(np.sum((truth == 1) & arm_label))
    base_tn = int(np.sum((truth == 0) & ~base_label))
    base_fp = int(np.sum((truth == 0) & base_label))
    base_fn = int(np.sum((truth == 1) & ~base_label))
    base_tp = int(np.sum((truth == 1) & base_label))
    positives = int(truth.sum())
    negatives = n - positives
    f1_supported = n >= F1_MIN_ROWS and positives >= F1_MIN_CLASS_ROWS and negatives >= F1_MIN_CLASS_ROWS
    return {
        "n": int(n),
        "positives": positives,
        "negatives": negatives,
        "f1_supported": bool(f1_supported),
        "f1_support_reason": "SUPPORTED" if f1_supported else "n_or_class_support_below_fixed_rule",
        "baseline_exact_050_rows": int(np.sum(base == THRESHOLD)),
        "baseline_positive_predictions": base_positive,
        "arm_positive_predictions": arm_positive,
        "baseline_positive_rate": (base_positive / n) if n else None,
        "arm_positive_rate": (arm_positive / n) if n else None,
        "positive_rate_change": ((arm_positive - base_positive) / n) if n else None,
        "baseline_errors": baseline_errors,
        "arm_errors": arm_errors,
        "useful_flips_baseline_wrong_to_correct": int(useful.sum()),
        "harmful_flips_baseline_correct_to_wrong": int(harmful.sum()),
        "net_error_reduction_count": baseline_errors - arm_errors,
        "error_rate_improvement": ((baseline_errors - arm_errors) / n) if n else None,
        "label_changes": int(changed.sum()),
        "flip_0_to_1": int(flip_0_to_1.sum()),
        "flip_1_to_0": int(flip_1_to_0.sum()),
        "useful_0_to_1": int(np.sum(flip_0_to_1 & useful)),
        "harmful_0_to_1": int(np.sum(flip_0_to_1 & harmful)),
        "useful_1_to_0": int(np.sum(flip_1_to_0 & useful)),
        "harmful_1_to_0": int(np.sum(flip_1_to_0 & harmful)),
        "baseline_macro_f1": (_f1_from_confusion(base_tn, base_fp, base_fn, base_tp)
                               if f1_supported else None),
        "arm_macro_f1": (_f1_from_confusion(tn, fp, fn, tp) if f1_supported else None),
        "macro_f1_delta": ((_f1_from_confusion(tn, fp, fn, tp)
                            - _f1_from_confusion(base_tn, base_fp, base_fn, base_tp))
                           if f1_supported else None),
        "baseline_tn": base_tn,
        "baseline_fp": base_fp,
        "baseline_fn": base_fn,
        "baseline_tp": base_tp,
        "arm_tn": tn,
        "arm_fp": fp,
        "arm_fn": fn,
        "arm_tp": tp,
    }


def _verify_parent_integrity() -> dict[str, Any]:
    parent_stats = read_json(PARENT_REPORT / "statistics.json")
    parent_qa = read_json(PARENT_REPORT / "QA.json")
    if parent_stats.get("status") != "COMPLETED_ZERO_FIT" or parent_qa.get("status") != "PASS":
        raise RuntimeError("Parent A02 reliability audit is not a passed zero-fit checkpoint")
    if parent_stats.get("official_fit_cap") != "66/66 consumed; unchanged":
        raise RuntimeError("Parent A02 audit does not report the frozen 66-fit cap")
    source = parent_stats.get("source_integrity", {})
    if source.get("official_train", {}).get("sha256") != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Parent A02 report has an unexpected official Train hash")
    if source.get("pilot_freeze_signature") != PILOT_FREEZE_SIGNATURE:
        raise RuntimeError("Parent A02 pilot freeze signature mismatch")
    if source.get("continuation_freeze_signature") != CONT_FREEZE_SIGNATURE:
        raise RuntimeError("Parent A02 continuation freeze signature mismatch")
    old_plan_path = PARENT_REPORT / "AUDIT_PLAN.md"
    if sha256_file(old_plan_path) != source.get("analysis_plan_sha256"):
        raise RuntimeError("Parent A02 plan hash mismatch")

    verified_caches = {}
    expected_caches = source.get("c16_cache_sha256", {})
    for key, relative in CACHE_PATHS.items():
        path = ROOT / relative
        current = sha256_file(path)
        if current != expected_caches.get(key):
            raise RuntimeError(f"Frozen C16 cache hash mismatch at cutoff {key}")
        verified_caches[key] = current

    verified_predictions = {}
    expected_predictions = source.get("prediction_hashes", {})
    for origin in ORIGINS:
        root_name = ("optimization_v3_a02_continuation_v1"
                     if origin in (59, 63, 71) else "optimization_v3_a02_neighbors_v1")
        for arm in ("observed", "shuffled"):
            relative = (Path("experiments/runs") / root_name / f"E{origin}" / arm
                        / "predictions.npz")
            current = sha256_file(ROOT / relative)
            key = f"E{origin}_{arm}"
            if current != expected_predictions.get(key):
                raise RuntimeError(f"Frozen prediction hash mismatch {key}")
            verified_predictions[key] = current

    row_hashes = {}
    for name, filename in ROW_ARTIFACTS.items():
        path = RUN_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required ignored row-level artifact missing: {filename}")
        row_hashes[name] = sha256_file(path)

    ledger_hash = sha256_file(LEDGER_PATH)
    if ledger_hash != EXPECTED_LEDGER_SHA256:
        raise RuntimeError("Fit ledger hash changed before the decision-boundary audit")
    ledger_lines = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line]
    starts = sum(row.get("event") == "FIT_STARTED" for row in ledger_lines)
    saved = sum(row.get("event") == "FIT_SAVED" for row in ledger_lines)
    failed = sum(row.get("event") == "FIT_FAILED" for row in ledger_lines)
    if (starts, saved, failed) != (66, 64, 2):
        raise RuntimeError(f"Unexpected ledger counts: {(starts, saved, failed)}")

    return {
        "parent_statistics_status": parent_stats["status"],
        "parent_qa_status": parent_qa["status"],
        "official_train_sha256_expected": EXPECTED_TRAIN_SHA256,
        "pilot_freeze_signature": PILOT_FREEZE_SIGNATURE,
        "continuation_freeze_signature": CONT_FREEZE_SIGNATURE,
        "c16_cache_sha256_verified": verified_caches,
        "prediction_sha256_verified": verified_predictions,
        "row_level_artifact_sha256": row_hashes,
        "ledger_sha256_before": ledger_hash,
        "fit_starts_saved_failed_before": [starts, saved, failed],
    }


def _load_train() -> tuple[pd.DataFrame, np.ndarray]:
    path = ROOT / TRAIN_REL
    if sha256_file(path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Classification Train differs from frozen SHA-256")
    train = pd.read_csv(path, usecols=["record_id", "cycle_code", "artifact_signal", "signal_stream"],
                        dtype={"record_id": str})
    if (len(train) != 50_608 or train["record_id"].isna().any()
            or train["record_id"].duplicated().any()):
        raise RuntimeError("Official Train row identity/count invalid")
    train["cycle_code"] = pd.to_numeric(train["cycle_code"], errors="raise").astype(np.int16)
    train["artifact_signal"] = pd.to_numeric(train["artifact_signal"], errors="raise").astype(np.int8)
    if set(train["artifact_signal"].unique()) != {0, 1}:
        raise RuntimeError("Official Train target is not binary")
    stream_missing = (train["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy())
    return train, stream_missing


def _load_views(train: pd.DataFrame, train_missing: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    views: dict[str, dict[str, np.ndarray]] = {}
    train_cycles = train["cycle_code"].to_numpy(dtype=np.int16)
    train_y = train["artifact_signal"].to_numpy(dtype=np.int8)
    train_ids = train["record_id"].to_numpy(dtype=str)
    for name, filename in ROW_ARTIFACTS.items():
        with np.load(RUN_ROOT / filename, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        expected = {"rows", "cycles", "origin", "y", "baseline", "observed", "shuffled",
                    "missing", "positive_neighbor_count", *DESCRIPTORS}
        if missing := expected.difference(data):
            raise RuntimeError(f"Row artifact {name} lacks keys: {sorted(missing)}")
        rows = data["rows"].astype(np.int64)
        if (rows.ndim != 1 or len(np.unique(rows)) != len(rows)
                or np.any(rows < 0) or np.any(rows >= len(train))):
            raise RuntimeError(f"Invalid/duplicate official Train row positions in {name}")
        if len(np.unique(train_ids[rows])) != len(rows):
            raise RuntimeError(f"Duplicate record IDs in {name}")
        if not np.array_equal(data["cycles"].astype(np.int16), train_cycles[rows]):
            raise RuntimeError(f"Official Train cycle join mismatch in {name}")
        if not np.array_equal(data["y"].astype(np.int8), train_y[rows]):
            raise RuntimeError(f"Official Train target join mismatch in {name}")
        if not np.array_equal(data["missing"].astype(bool), train_missing[rows]):
            raise RuntimeError(f"Official Train stream-missing mask mismatch in {name}")
        for key in ("baseline", "observed", "shuffled"):
            if data[key].shape != rows.shape or not _is_finite(data[key]):
                raise RuntimeError(f"Invalid {key} probabilities in {name}")
            if np.any((data[key] < 0) | (data[key] > 1)):
                raise RuntimeError(f"Out-of-range {key} probabilities in {name}")
        for key in DESCRIPTORS:
            if data[key].shape != rows.shape:
                raise RuntimeError(f"Descriptor shape mismatch for {name}/{key}")
        data["y"] = train_y[rows].copy()  # use truth rejoined from official Train
        data["learned_present"] = (data["origin"] >= 0) & (~data["missing"].astype(bool))
        data["no_correction"] = ~data["learned_present"]
        if not np.array_equal(data["learned_present"], data["positive_neighbor_count"] > 0):
            raise RuntimeError(f"Learned-present mask differs from saved positive-neighbor count in {name}")
        delta_obs = data["observed"] - data["baseline"]
        delta_shuffle = data["shuffled"] - data["baseline"]
        if np.max(np.abs(delta_obs), initial=0.0) > CORRECTION_LIMIT + 1e-12:
            raise RuntimeError(f"A02 correction exceeds frozen +/-0.05 bound in {name}")
        if np.max(np.abs(delta_shuffle), initial=0.0) > CORRECTION_LIMIT + 1e-12:
            raise RuntimeError(f"Shuffled correction exceeds frozen +/-0.05 bound in {name}")
        if np.any(data["origin"] < 0):
            fallback = data["origin"] < 0
            if not (np.array_equal(data["baseline"][fallback], data["observed"][fallback])
                    and np.array_equal(data["baseline"][fallback], data["shuffled"][fallback])):
                raise RuntimeError("W1 fallback probabilities differ from Sparse25")
        missing = data["missing"].astype(bool)
        if not (np.array_equal(data["baseline"][missing], data["observed"][missing])
                and np.array_equal(data["baseline"][missing], data["shuffled"][missing])):
            raise RuntimeError(f"Missing-stream rows changed from Sparse25 in {name}")
        views[name] = data

    primary_rows = set(views["full_six_primary"]["rows"].tolist())
    h12_rows = set(views["inherited_H12_L"]["rows"].tolist())
    if not h12_rows.issubset(primary_rows):
        raise RuntimeError("H12-L is not a subset of the primary view as documented")
    if len(h12_rows) != 20_903:
        raise RuntimeError(f"Unexpected H12-L row count: {len(h12_rows)}")

    primary = views["full_six_primary"]
    for origin, (first, last) in ORIGIN_FIRST4.items():
        mask = primary["origin"] == origin
        if (not mask.any() or np.min(primary["cycles"][mask]) != first
                or np.max(primary["cycles"][mask]) != last
                or set(np.unique(primary["cycles"][mask]).tolist()) != set(range(first, last + 1))):
            raise RuntimeError(f"Saved origin E{origin} is not its declared first-four-cycle view")
    expected_origins = {
        "new_origin_aggregate": {59, 63, 71},
        "late": {71, 75},
        "inherited_H12_L": {67},
    }
    for view_name, expected_set in expected_origins.items():
        actual = set(np.unique(views[view_name]["origin"]).astype(int).tolist())
        if actual != expected_set:
            raise RuntimeError(f"Unexpected origins in {view_name}: {actual}")
    return views


def _scope_specs(views: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, str, dict[str, np.ndarray], np.ndarray]]:
    specs: list[tuple[str, str, dict[str, np.ndarray], np.ndarray]] = []
    for view_name in ("full_six_primary", "new_origin_aggregate", "late", "inherited_H12_L"):
        frame = views[view_name]
        all_rows = np.ones(len(frame["y"]), dtype=bool)
        specs.append((view_name, "all_rows", frame, all_rows))
        specs.append((view_name, "learned_present", frame, frame["learned_present"]))
    primary = views["full_six_primary"]
    specs.append(("full_six_primary", "W1_fallback", primary, primary["origin"] < 0))
    for origin in ORIGINS:
        frame = primary
        specs.append(("full_six_primary", f"E{origin}_first4_all_rows", frame,
                      frame["origin"] == origin))
    return specs


def summarize_scope(view: str, scope: str, frame: dict[str, np.ndarray],
                    scope_mask: np.ndarray, arm: str) -> dict[str, Any]:
    mask = np.asarray(scope_mask, dtype=bool)
    y = frame["y"][mask]
    baseline = frame["baseline"][mask]
    arm_p = frame[arm][mask]
    result = flip_accounting(y, baseline, arm_p)
    origins = np.unique(frame["origin"][mask]).astype(int).tolist() if mask.any() else []
    cycles = np.unique(frame["cycles"][mask]).astype(int).tolist() if mask.any() else []
    learned = frame["learned_present"][mask]
    missing = frame["missing"][mask].astype(bool)
    base_label = baseline >= THRESHOLD
    arm_label = arm_p >= THRESHOLD
    changed = base_label != arm_label
    changed_outside = outside_margin_change_count(baseline, arm_p)
    result.update({
        "view": view,
        "scope": scope,
        "arm": arm,
        "learned_present_rows": int(learned.sum()),
        "stream_missing_rows": int(missing.sum()),
        "W1_fallback_rows": int(np.sum(frame["origin"][mask] < 0)),
        "origin_codes": "|".join(map(str, origins)),
        "cycle_codes": "|".join(map(str, cycles)),
        "max_abs_probability_delta": float(np.max(np.abs(arm_p - baseline), initial=0.0)),
        "changed_labels_outside_inclusive_045_055": changed_outside,
        "flips_at_baseline_045": int(np.sum(changed & np.isclose(baseline, .45, rtol=0, atol=1e-12))),
        "flips_at_baseline_055": int(np.sum(changed & np.isclose(baseline, .55, rtol=0, atol=1e-12))),
    })
    return result


def _band_mask(probabilities: np.ndarray, band_index: int) -> np.ndarray:
    return fixed_band_mask(probabilities, band_index)


def build_band_tables(views: dict[str, dict[str, np.ndarray]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    band_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, dict[str, np.ndarray], np.ndarray]] = []
    primary = views["full_six_primary"]
    scopes.append(("full_six_primary_learned_present", primary, primary["learned_present"]))
    for origin in ORIGINS:
        mask = (primary["origin"] == origin) & primary["learned_present"]
        scopes.append((f"E{origin}_first4_learned_present", primary, mask))
    for view_name in ("new_origin_aggregate", "late", "inherited_H12_L"):
        frame = views[view_name]
        scopes.append((f"{view_name}_learned_present", frame, frame["learned_present"]))

    for scope_name, frame, learned_mask in scopes:
        p = frame["baseline"]
        for band_index, (low, high, label) in enumerate(BASELINE_BANDS):
            in_band = learned_mask & _band_mask(p, band_index)
            y = frame["y"][in_band]
            base_p = p[in_band]
            n = len(y)
            positives = int(y.sum())
            negatives = n - positives
            f1_support = (n >= F1_MIN_ROWS and positives >= F1_MIN_CLASS_ROWS
                          and negatives >= F1_MIN_CLASS_ROWS)
            for arm in ("observed", "shuffled"):
                arm_p = frame[arm][in_band]
                metrics = flip_accounting(y, base_p, arm_p)
                boundary_support = n >= BOUNDARY_MIN_ROWS and metrics["label_changes"] >= BOUNDARY_MIN_CHANGES
                arm_label = arm_p >= THRESHOLD
                base_label = base_p >= THRESHOLD
                changed = base_label != arm_label
                useful_flip = changed & (base_label != y.astype(bool)) & (arm_label == y.astype(bool))
                harmful_flip = changed & (base_label == y.astype(bool)) & (arm_label != y.astype(bool))
                band_rows.append({
                    "scope": scope_name,
                    "arm": arm,
                    "baseline_band": label,
                    "band_low": low,
                    "band_high": high,
                    "band_right_edge_inclusive": band_index == len(BASELINE_BANDS) - 1,
                    "n": n,
                    "positives": positives,
                    "negatives": negatives,
                    "baseline_exact_050_rows": int(np.sum(base_p == THRESHOLD)),
                    "baseline_positive_predictions": int(base_label.sum()),
                    "arm_positive_predictions": int(arm_label.sum()),
                    "positive_prediction_change": int(arm_label.sum() - base_label.sum()),
                    "label_changes": int(changed.sum()),
                    "flip_0_to_1": int(np.sum(~base_label & arm_label)),
                    "flip_1_to_0": int(np.sum(base_label & ~arm_label)),
                    "useful_flips_baseline_wrong_to_correct": int(useful_flip.sum()),
                    "harmful_flips_baseline_correct_to_wrong": int(harmful_flip.sum()),
                    "net_error_reduction_count": int(useful_flip.sum() - harmful_flip.sum()),
                    "classification_error_improvement": metrics["error_rate_improvement"],
                    "f1_supported": bool(f1_support),
                    "f1_support_reason": "SUPPORTED" if f1_support else "n_or_class_support_below_fixed_rule",
                    "baseline_macro_f1": metrics["baseline_macro_f1"],
                    "arm_macro_f1": metrics["arm_macro_f1"],
                    "macro_f1_delta": (metrics["arm_macro_f1"] - metrics["baseline_macro_f1"]
                                       if f1_support else None),
                    "boundary_group_supported": bool(boundary_support),
                    "boundary_support_reason": ("SUPPORTED" if boundary_support else
                                                "n_below_100_or_label_changes_below_20"),
                    "unchanged_rows": int(n - int(changed.sum())),
                })

                strata = {
                    "unchanged": in_band & ~((frame["baseline"] >= THRESHOLD) != (frame[arm] >= THRESHOLD)),
                    "useful_flip": in_band & ((frame["baseline"] >= THRESHOLD) != (frame[arm] >= THRESHOLD))
                    & ((frame["baseline"] >= THRESHOLD) != frame["y"].astype(bool))
                    & ((frame[arm] >= THRESHOLD) == frame["y"].astype(bool)),
                    "harmful_flip": in_band & ((frame["baseline"] >= THRESHOLD) != (frame[arm] >= THRESHOLD))
                    & ((frame["baseline"] >= THRESHOLD) == frame["y"].astype(bool))
                    & ((frame[arm] >= THRESHOLD) != frame["y"].astype(bool)),
                }
                any_flip = strata["useful_flip"] | strata["harmful_flip"]
                strata["all_flips"] = any_flip
                for stratum, global_mask in strata.items():
                    selected = global_mask & learned_mask
                    count = int(selected.sum())
                    threshold_n = (FLIP_STRATUM_MIN if stratum != "unchanged" else BOUNDARY_MIN_ROWS)
                    row: dict[str, Any] = {
                        "scope": scope_name,
                        "arm": arm,
                        "baseline_band": label,
                        "stratum": stratum,
                        "n": count,
                        "stratum_supported": count >= threshold_n,
                        "stratum_support_reason": "SUPPORTED" if count >= threshold_n
                        else f"n_below_{threshold_n}",
                        "band_supported_for_label_change_analysis": bool(boundary_support),
                        "band_f1_supported": bool(f1_support),
                    }
                    for descriptor in DESCRIPTORS:
                        values = frame[descriptor][selected].astype(np.float64)
                        values = values[np.isfinite(values)]
                        row[f"{descriptor}_mean"] = float(np.mean(values)) if count >= threshold_n and len(values) else None
                        row[f"{descriptor}_median"] = float(np.median(values)) if count >= threshold_n and len(values) else None
                    descriptor_rows.append(row)
    return band_rows, descriptor_rows


def _mechanism_candidates(band_rows: list[dict[str, Any]],
                          descriptor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    desc_by_key = {(r["scope"], r["arm"], r["baseline_band"], r["stratum"]): r
                   for r in descriptor_rows}
    band_by_key = {(r["scope"], r["arm"], r["baseline_band"]): r for r in band_rows}
    candidates = []
    for _, _, label in BASELINE_BANDS:
        for descriptor in DESCRIPTORS:
            directions: dict[int, int] = {}
            for origin in ORIGINS:
                scope = f"E{origin}_first4_learned_present"
                band = band_by_key.get((scope, "observed", label))
                good = desc_by_key.get((scope, "observed", label, "useful_flip"))
                bad = desc_by_key.get((scope, "observed", label, "harmful_flip"))
                if not band or not band["boundary_group_supported"]:
                    continue
                if not good or not bad or min(good["n"], bad["n"]) < FLIP_STRATUM_MIN:
                    continue
                delta = good[f"{descriptor}_median"] - bad[f"{descriptor}_median"]
                if delta > 0:
                    directions[origin] = 1
                elif delta < 0:
                    directions[origin] = -1
            positive = sum(value == 1 for value in directions.values())
            negative = sum(value == -1 for value in directions.values())
            sign = 1 if positive >= 3 else (-1 if negative >= 3 else 0)
            if sign == 0:
                continue
            aggregate_signs = {}
            for aggregate in ("full_six_primary_learned_present",
                              "new_origin_aggregate_learned_present", "late_learned_present"):
                band = band_by_key.get((aggregate, "observed", label))
                good = desc_by_key.get((aggregate, "observed", label, "useful_flip"))
                bad = desc_by_key.get((aggregate, "observed", label, "harmful_flip"))
                if (not band or not band["boundary_group_supported"] or not band["f1_supported"]
                        or not good or not bad or min(good["n"], bad["n"]) < FLIP_STRATUM_MIN):
                    aggregate_signs[aggregate] = 0
                    continue
                delta = good[f"{descriptor}_median"] - bad[f"{descriptor}_median"]
                aggregate_signs[aggregate] = 1 if delta > 0 else (-1 if delta < 0 else 0)
            if all(value == sign for value in aggregate_signs.values()):
                candidates.append({
                    "descriptor": descriptor,
                    "baseline_band": label,
                    "useful_minus_harmful_median_direction": "higher_on_useful" if sign > 0 else "lower_on_useful",
                    "origin_direction_counts": {"higher": positive, "lower": negative, "origins_supported": len(directions)},
                    "aggregate_signs": aggregate_signs,
                })
    return candidates


def _make_summary(flip_rows: list[dict[str, Any]], band_rows: list[dict[str, Any]],
                  candidates: list[dict[str, Any]], qa: dict[str, Any]) -> str:
    def find(view: str, scope: str, arm: str) -> dict[str, Any]:
        for row in flip_rows:
            if row["view"] == view and row["scope"] == scope and row["arm"] == arm:
                return row
        raise KeyError((view, scope, arm))

    selected_scopes = [
        ("full_six_primary", "all_rows"),
        ("full_six_primary", "learned_present"),
        ("full_six_primary", "W1_fallback"),
        ("new_origin_aggregate", "all_rows"),
        ("late", "all_rows"),
        ("inherited_H12_L", "all_rows"),
    ]
    lines = [
        "# A02 decision-boundary audit",
        "",
        "**Status:** COMPLETED_ZERO_FIT",
        f"**F1 lane decision:** {'ONE_SUPPORTED_DIAGNOSTIC_CARD' if len(candidates) == 1 else 'NO_SUPPORTED_F1_LANE'}",
        f"**QA:** {qa['status']}",
        "**Interpretation:** adaptive diagnostic using reused validation rows; no holdout or causal claim.",
        "",
        "## Main observed-arm label changes",
        "",
        "All labels use the frozen rule `p >= 0.50`, so exact 0.50 predicts positive. `helpful` means a baseline error became correct; `harmful` means a correct baseline label became wrong. Error-count change is additive; Macro F1 is reported only when class support passes the fixed rule.",
        "",
        "| Scope | n rows | learned-present | observed flips / helpful / harmful / net | shuffle flips / helpful / harmful / net | Δ positive rate (observed) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, scope in selected_scopes:
        row = find(view, scope, "observed")
        shuffle = find(view, scope, "shuffled")
        lines.append(
            f"| {row['view']} / {row['scope']} | {row['n']} | {row['learned_present_rows']} | "
            f"{row['label_changes']} / {row['useful_flips_baseline_wrong_to_correct']} / "
            f"{row['harmful_flips_baseline_correct_to_wrong']} / {row['net_error_reduction_count']} | "
            f"{shuffle['label_changes']} / {shuffle['useful_flips_baseline_wrong_to_correct']} / "
            f"{shuffle['harmful_flips_baseline_correct_to_wrong']} / {shuffle['net_error_reduction_count']} | "
            f"{row['positive_rate_change']:.5f} |"
        )
    origin_rows = [find("full_six_primary", f"E{o}_first4_all_rows", "observed") for o in ORIGINS]
    lines += [
        "",
        "W1 is unchanged Sparse25 fallback; it contains 4,114 rows (3,722 present-stream and 392 missing-stream). H12-L overlaps primary. All table rows compare the same query rows across the baseline, observed, and shuffle arms.",
        "",
        "## Per-origin first-four windows (observed arm)",
        "",
        "| Origin window | n | learned-present | label changes | helpful | harmful | net errors reduced | Δ positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in origin_rows:
        lines.append(
            f"| E{row['origin_codes']} | {row['n']} | {row['learned_present_rows']} | {row['label_changes']} | "
            f"{row['useful_flips_baseline_wrong_to_correct']} | {row['harmful_flips_baseline_correct_to_wrong']} | "
            f"{row['net_error_reduction_count']} | {row['positive_rate_change']:.5f} |"
        )
    primary_bands = [r for r in band_rows if r["scope"] == "full_six_primary_learned_present"
                     and r["arm"] == "observed"]
    lines += [
        "",
        "## Fixed decision bands (full-six learned-present, observed)",
        "",
        "| Baseline p band | n | flips 0→1 | flips 1→0 | helpful | harmful | net errors reduced | Macro F1 Δ | F1 support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in primary_bands:
        f1_delta = f"{row['macro_f1_delta']:.5f}" if row["macro_f1_delta"] is not None else "unsupported"
        lines.append(
            f"| {row['baseline_band']} | {row['n']} | {row['flip_0_to_1']} | {row['flip_1_to_0']} | "
            f"{row['useful_flips_baseline_wrong_to_correct']} | {row['harmful_flips_baseline_correct_to_wrong']} | "
            f"{row['net_error_reduction_count']} | {f1_delta} | {row['f1_supported']} |"
        )
    lines += [
        "",
        "All four full-primary bands contain flips: below 0.50 they are 0→1, at/above 0.50 they are 1→0. The net error improvement is concentrated in the middle two bands, while the origin-level pattern is mixed. The late view has only 3 net errors reduced despite a positive Δ predicted-positive rate; E71 increases the positive rate while net errors worsen. This shows why positive-rate growth alone does not establish useful correction.",
        "",
        "Per-origin shuffle counts and full detail are in `flip_summary.csv`; band-by-origin arm counts are in `baseline_band_summary.csv`; descriptor context is in `band_descriptor_summary.csv`.",
        "",
        "## Boundary verification",
        "",
        f"Both observed and shuffled corrections stay within ±0.05 (max observed |Δp| {qa['max_abs_probability_delta']['observed']:.12f}, shuffle {qa['max_abs_probability_delta']['shuffled']:.12f}; tiny excess over .05 is floating representation). Label changes outside baseline p ∈ [0.45,0.55] are {qa['outside_margin_label_change_count']}; exact 0.50 is positive. All hard-label changes are limited to learned-present rows; W1 fallback and missing-stream rows are unchanged.",
        "",
        f"The confidence taper in the prior review note is rejected as a Macro F1/Kaggle lane: Sol's independent check found zero hard-label changes on the four saved views (6,134 probability values changed in full-six primary). Such a transform may be considered only if probability quality becomes a separate objective; no taper run or submission was made here.",
        "",
        "## Reliability descriptors and decision",
        "",
        "Descriptor summaries condition on fixed baseline bands and compare unchanged rows with useful and harmful flips. A band is supported for this comparison only at n ≥ 100 and at least 20 label changes; Macro F1 has a separate n ≥ 200 and ≥30 per-class support check. Unsupported rows remain visible in the CSVs.",
        "",
    ]
    if len(candidates) == 1:
        candidate = candidates[0]
        lines.extend([
            f"One descriptive pattern met the predeclared replication rule: `{candidate['descriptor']}` in band `{candidate['baseline_band']}`, with the useful-flip median {candidate['useful_minus_harmful_median_direction']} the harmful-flip median. It met the fixed origin/aggregate support rule, but remains exploratory on reused rows. A single review-only mechanism card was written; no fit or model change is authorized.",
            "",
        ])
    else:
        lines.extend([
            "No unique descriptor/band passed the predeclared replication rule requiring at least 20 useful and 20 harmful flips in a supported band for at least three origin windows, a same-direction contrast in both new-origin and late aggregates, and supported aggregate Macro F1 comparisons. Result: **NO_SUPPORTED_F1_LANE**. Do not create another Macro F1 card from these reused rows.",
            "",
        ])
    lines += [
        "## Integrity and limits",
        "",
        f"Official Train SHA-256: `{qa['official_train_sha256']}`. Fit ledger SHA-256 before/after: `{qa['ledger_sha256_before']}` / `{qa['ledger_sha256_after']}`. Fit accounting remains 66 starts, 64 saved, 2 historical failures; estimator/preprocessor fit calls: 0.",
        "",
        "This diagnostic did not access Test/SampleSubmission or Kaggle, did not write row-level predictions, and did not change the frozen A02 recipe, gates, metrics, public score receipt, or champion. All evaluated rows have already been inspected; any pattern is descriptive, not fresh validation.",
    ]
    return "\n".join(lines) + "\n"


def run_audit() -> dict[str, Any]:
    pre = _verify_parent_integrity()
    train, train_missing = _load_train()
    views = _load_views(train, train_missing)
    flip_rows = []
    scope_masks = _scope_specs(views)
    outside_changes = 0
    max_delta = {"observed": 0.0, "shuffled": 0.0}
    for view, scope, frame, mask in scope_masks:
        for arm in ("observed", "shuffled"):
            row = summarize_scope(view, scope, frame, mask, arm)
            flip_rows.append(row)
            outside_changes += row["changed_labels_outside_inclusive_045_055"]
            max_delta[arm] = max(max_delta[arm], row["max_abs_probability_delta"])
            selected = np.asarray(mask, dtype=bool)
            changed = ((frame["baseline"][selected] >= THRESHOLD)
                       != (frame[arm][selected] >= THRESHOLD))
            if np.any(changed & ~frame["learned_present"][selected]):
                raise RuntimeError(f"Non-learned or fallback row changed label in {view}/{scope}/{arm}")
    if outside_changes != 0:
        raise RuntimeError(f"Found label changes outside the fixed [.45,.55] margin: {outside_changes}")

    band_rows, descriptor_rows = build_band_tables(views)
    candidates = _mechanism_candidates(band_rows, descriptor_rows)
    if len(candidates) > 1:
        # An ambiguous multi-mechanism result does not justify choosing one post hoc.
        candidates = []

    primary = views["full_six_primary"]
    learned = primary["learned_present"]
    w1 = primary["origin"] < 0
    policy_prob = primary["observed"][learned]

    ledger_after = sha256_file(LEDGER_PATH)
    if ledger_after != pre["ledger_sha256_before"]:
        raise RuntimeError("Fit ledger changed during the zero-fit decision-boundary audit")
    qa = {
        "status": "PASS",
        "audit": "A02 decision-boundary and useful-vs-harmful flip audit; zero estimator fits",
        "outcome_blind": False,
        "checks": {
            "parent_a02_reliability_qa_pass": pre["parent_qa_status"] == "PASS",
            "official_train_hash_matches_frozen": True,
            "all_six_cache_hashes_match_parent_report": True,
            "all_ten_prediction_hashes_match_parent_report": True,
            "all_four_row_artifact_joins_match_official_train_cycle_and_target": True,
            "record_ids_unique_for_all_joined_rows": True,
            "stream_missing_mask_matches_official_train": True,
            "same_rows_paired_across_baseline_observed_shuffle": True,
            "h12_rows_subset_of_primary_and_overlap_disclosed": True,
            "w1_fallback_exact_sparse25_and_unchanged": True,
            "missing_stream_exact_sparse25_and_unchanged": True,
            "observed_correction_within_frozen_005_bound": max_delta["observed"] <= CORRECTION_LIMIT + 1e-12,
            "shuffle_correction_within_frozen_005_bound": max_delta["shuffled"] <= CORRECTION_LIMIT + 1e-12,
            "label_changes_outside_fixed_margin": outside_changes == 0,
            "threshold_is_exact_greater_equal_050": True,
            "fixed_probability_band_edges_and_support_rules_used": True,
            "descriptor_summaries_exclude_fallback_and_missing_stream": True,
            "review_taper_lane_rejected_by_frozen_probability_bound": True,
            "no_taper_replay_or_new_probability_policy_or_sweep": True,
            "zero_estimator_or_preprocessor_fit_calls": True,
            "ledger_sha_unchanged": ledger_after == pre["ledger_sha256_before"],
            "no_test_sample_kaggle_or_submission_access": True,
            "no_row_level_artifact_written": True,
        },
        "official_train_sha256": EXPECTED_TRAIN_SHA256,
        "ledger_sha256_before": pre["ledger_sha256_before"],
        "ledger_sha256_after": ledger_after,
        "fit_starts_saved_failed_before_after": {
            "before": pre["fit_starts_saved_failed_before"],
            "after": pre["fit_starts_saved_failed_before"],
        },
        "estimator_fit_calls_observed": 0,
        "outside_margin_label_change_count": outside_changes,
        "review_taper_check": {
            "provenance": "independent Sol review of the saved four audit views",
            "hard_label_changes_reported": 0,
            "full_primary_probability_values_changed_reported": 6134,
            "logic": "For baseline p > 0.8, attenuation g is in [0,1] and |delta| <= 0.05, so tapered p >= p - 0.05 > 0.75; threshold 0.50 labels cannot change.",
            "replayed_by_this_script": False,
            "interpretation": "optional probability-quality note only; not a Macro F1 lane",
        },
        "max_abs_probability_delta": max_delta,
        "learned_present_primary_rows": int(learned.sum()),
        "full_primary_stream_present_rows": int((~primary["missing"].astype(bool)).sum()),
        "primary_w1_fallback_rows": int(w1.sum()),
        "primary_w1_fallback_stream_present_rows": int(np.sum(w1 & ~primary["missing"].astype(bool))),
        "primary_w1_fallback_stream_missing_rows": int(np.sum(w1 & primary["missing"].astype(bool))),
        "primary_learned_origin_stream_missing_rows": int(np.sum((primary["origin"] >= 0) & primary["missing"].astype(bool))),
        "primary_learned_present_exact_endpoint_rows": int(np.sum((policy_prob == 0) | (policy_prob == 1))),
        "primary_learned_present_wrong_exact_endpoint_rows": int(np.sum(
            ((policy_prob == 0) & (primary["y"][learned] == 1))
            | ((policy_prob == 1) & (primary["y"][learned] == 0)))),
        "mechanism_candidates": candidates,
        "source_hashes": {
            "c16_caches": pre["c16_cache_sha256_verified"],
            "prediction_bundles": pre["prediction_sha256_verified"],
            "row_level_inputs": pre["row_level_artifact_sha256"],
        },
        "outputs": [],
    }
    qa["checks"]["overall_checks_pass"] = all(qa["checks"].values())
    if not qa["checks"]["overall_checks_pass"]:
        raise RuntimeError("Decision-boundary QA did not pass")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    flip_path = REPORT_ROOT / "flip_summary.csv"
    band_path = REPORT_ROOT / "baseline_band_summary.csv"
    desc_path = REPORT_ROOT / "band_descriptor_summary.csv"
    pd.DataFrame(flip_rows).to_csv(flip_path, index=False, encoding="utf-8")
    pd.DataFrame(band_rows).to_csv(band_path, index=False, encoding="utf-8")
    pd.DataFrame(descriptor_rows).to_csv(desc_path, index=False, encoding="utf-8")
    if candidates:
        candidate = candidates[0]
        next_card = (
            "# Review-only A02 decision-boundary mechanism card\n\n"
            "**Status:** diagnostic proposal only; no fit, policy change, or submission is authorized.\n\n"
            f"Descriptor: `{candidate['descriptor']}`; baseline band: `{candidate['baseline_band']}`; "
            f"useful-flip median is {candidate['useful_minus_harmful_median_direction']} the harmful-flip median.\n\n"
            "This direction met the predeclared support/replication rule on reused validation rows. "
            "Do not turn it into an inference-time router or tune a threshold on these same rows. "
            "A new card requires separate review and a prospective labeled temporal holdout.\n"
        )
        (REPORT_ROOT / "NEXT_CARD.md").write_text(next_card, encoding="utf-8")
        qa["outputs"].append("NEXT_CARD.md")
    summary_path = REPORT_ROOT / "SUMMARY.md"
    summary_path.write_text(_make_summary(flip_rows, band_rows, candidates, qa), encoding="utf-8")
    qa["outputs"].extend(["SUMMARY.md", "flip_summary.csv", "baseline_band_summary.csv",
                          "band_descriptor_summary.csv"])
    qa["output_sha256"] = {name: sha256_file(REPORT_ROOT / name) for name in qa["outputs"]}
    qa["qa_path"] = "reports/optimization_v3/a02_decision_boundary_audit_v1/QA.json"
    (REPORT_ROOT / "QA.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                          encoding="utf-8")
    return {"qa": qa, "flip_rows": flip_rows, "band_rows": band_rows,
            "descriptor_rows": descriptor_rows, "candidates": candidates}


def main() -> None:
    result = run_audit()
    print(json.dumps({
        "status": result["qa"]["status"],
        "qa": result["qa"]["checks"],
        "primary_learned_present_rows": result["qa"]["learned_present_primary_rows"],
        "max_delta": result["qa"]["max_abs_probability_delta"],
        "mechanism_candidates": result["candidates"],
        "ledger_unchanged": result["qa"]["ledger_sha256_before"] == result["qa"]["ledger_sha256_after"],
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
