"""Independently verify the four-fit sparse regression pilot without fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import sha256 as file_sha256
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import _fit_fingerprint
from src.utils.paths import task_files

FREEZE_PATH = ROOT / "experiments" / "optimization_v2" / "regression_repr" / "freeze.json"
START_PATH = ROOT / "experiments" / "optimization_v2" / "regression_repr" / "pilot_started.json"
PILOT_PATH = ROOT / "reports" / "optimization_v2" / "regression_repr" / "pilot.json"
PREDICTION_PATH = ROOT / "experiments" / "runs" / "optimization_v2_regression_repr" / "pilot_predictions.npz"
VERIFICATION_PATH = ROOT / "reports" / "optimization_v2" / "regression_repr" / "verification.json"
BASELINE_REPORT = ROOT / "reports" / "dual_campaign" / "regression" / "stage1_paired_blend.json"
BASELINE_CACHE = ROOT / "experiments" / "runs" / "dual_reg_v1" / "stage1_paired_blends.npz"
ROUND2_PROTOCOL = ROOT / "experiments" / "regression_round2" / "protocol.json"
VIEWS = {"W1": (55, 56, 59), "W4": (67, 68, 71), "H12_E": (55, 56, 67), "H12_L": (67, 68, 79)}
RECIPES = ("sgd_l1", "lgbm_sparse_l1")


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_hash(value: Any, dtype: str = "<i8") -> str:
    array = np.asarray(value, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {
        "n": int(diff.size),
        "mae": float(np.abs(diff).sum() / diff.size) if diff.size else None,
        "signed_bias_prediction_minus_target": float(diff.sum() / diff.size) if diff.size else None,
        "over_rate": float(np.count_nonzero(diff > 1e-10) / diff.size) if diff.size else None,
        "under_rate": float(np.count_nonzero(diff < -1e-10) / diff.size) if diff.size else None,
        "absolute_error_p95": float(np.quantile(np.abs(diff), .95)) if diff.size else None,
    }


def _cycle_rank(raw: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    # Independent implementation of average percentile ranks, with ties kept.
    prediction = np.asarray(raw, dtype=float)
    cycles = np.asarray(cycle)
    mapped = np.empty(prediction.size, dtype=float)
    for value in pd.unique(cycles):
        positions = np.flatnonzero(cycles == value)
        order = np.argsort(prediction[positions], kind="mergesort")
        sorted_values = prediction[positions][order]
        start = 0
        rank = np.empty(len(positions), dtype=float)
        while start < len(positions):
            stop = start + 1
            while stop < len(positions) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            average_rank = ((start + 1) + stop) / 2.0
            rank[order[start:stop]] = average_rank
            start = stop
        mapped[positions] = 100.0 * rank / len(positions)
    return mapped


def _group_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    text = frame["signal_stream"]
    missing = text.isna().to_numpy() | text.map(
        lambda value: isinstance(value, str) and not value.strip()
    ).to_numpy(dtype=bool)
    friction = pd.to_numeric(frame["entry_friction"], errors="coerce").to_numpy(dtype=float)
    masks = {
        "tail_le5": y <= 5.0,
        "tail_ge95": y >= 95.0,
        "EF0": friction == 0.0,
        "EF1": friction == 1.0,
        "stream_missing": missing,
        "stream_present": ~missing,
    }
    return {name: _metric(y[selector], pred[selector]) for name, selector in masks.items()}


def _close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def verify() -> None:
    for path in (FREEZE_PATH, START_PATH, PILOT_PATH, PREDICTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required pilot artifact missing: {path}")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
    started = json.loads(START_PATH.read_text(encoding="utf-8"))
    if started["freeze_sha256"] != file_sha256(FREEZE_PATH) or pilot["freeze_sha256"] != file_sha256(FREEZE_PATH):
        raise AssertionError("Pilot does not point to the exact pre-fit freeze receipt")
    if freeze["source_head"] != subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip():
        raise AssertionError("Repository HEAD no longer matches frozen source commit")
    for rel, digest in freeze["source_sha256"].items():
        if file_sha256(ROOT / rel) != digest:
            raise AssertionError(f"Frozen code/protocol changed: {rel}")
    if file_sha256(ROUND2_PROTOCOL) != freeze["round2_protocol_sha256"]:
        raise AssertionError("Temporal fold protocol changed")
    if file_sha256(BASELINE_REPORT) != freeze["current_ridge25"]["report_sha256"]:
        raise AssertionError("Ridge25 source report changed")
    if file_sha256(BASELINE_CACHE) != freeze["current_ridge25"]["cache_sha256"]:
        raise AssertionError("Ridge25 prediction cache changed")
    if file_sha256(PREDICTION_PATH) != pilot["prediction_cache_sha256"]:
        raise AssertionError("Candidate prediction cache hash mismatch")
    if pilot["fit_count"] != 4 or pilot["full_six_fit_continuation_performed"]:
        raise AssertionError("Pilot fit count/scope does not match the predeclared four-fit stop")
    if pilot["new_control_fits"] != 0 or pilot["test_rows_or_labels_used"]:
        raise AssertionError("Unexpected control fit or Test access")

    train_path = task_files("regression")["train"]
    if train_path is None or file_sha256(train_path) != freeze["official_train"]["sha256"]:
        raise AssertionError("Official Train bytes differ from the frozen source")
    frame = pd.read_csv(train_path, low_memory=False)
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=float)
    if len(frame) != freeze["official_train"]["rows"]:
        raise AssertionError("Official Train row count changed")

    baseline_data = np.load(BASELINE_CACHE, allow_pickle=False)
    candidate_data = np.load(PREDICTION_PATH, allow_pickle=False)
    expected_candidate_keys = {
        f"{prefix}_{recipe}_{cutoff}"
        for prefix in ("rows", "raw")
        for recipe in RECIPES
        for cutoff in (55, 67)
    }
    if set(candidate_data.files) != expected_candidate_keys:
        raise AssertionError("Prediction cache contains missing or unexpected fit arrays")

    fit_receipts: dict[str, Any] = {}
    for cutoff in (55, 67):
        fit_rows_expected = np.flatnonzero(cycle <= cutoff).astype(np.int64)
        val_low, val_high = (56, 67) if cutoff == 55 else (68, 79)
        val_rows_expected = np.flatnonzero((cycle >= val_low) & (cycle <= val_high)).astype(np.int64)
        if not np.array_equal(baseline_data[f"rows_{cutoff}"], val_rows_expected):
            raise AssertionError(f"Ridge25 baseline row order mismatch at {cutoff}")
        for recipe in RECIPES:
            key = f"{recipe}_{cutoff}"
            receipt = pilot["fits"][key]
            candidate_rows = np.asarray(candidate_data[f"rows_{key}"], dtype=np.int64)
            candidate_raw = np.asarray(candidate_data[f"raw_{key}"], dtype=np.float64)
            if not np.array_equal(candidate_rows, val_rows_expected):
                raise AssertionError(f"Candidate validation positions mismatch at {key}")
            if candidate_raw.shape != candidate_rows.shape or not np.isfinite(candidate_raw).all():
                raise AssertionError(f"Candidate prediction vector invalid at {key}")
            if _array_hash(candidate_rows) != receipt["validation_positions_sha256"]:
                raise AssertionError(f"Validation position digest mismatch at {key}")
            if _array_hash(candidate_raw, "<f8") != receipt["raw_prediction_sha256"]:
                raise AssertionError(f"Raw prediction digest mismatch at {key}")
            state = receipt["fit_state"]
            if state["fit_rows"] != len(fit_rows_expected) or state["input_feature_names"] != list(HYBRID_FEATURES):
                raise AssertionError(f"Fit receipt dimensions/features differ at {key}")
            if receipt["fit_positions_sha256"] != _array_hash(fit_rows_expected):
                raise AssertionError(f"Fit row identity mismatch at {key}")
            if receipt["fit_cycle_max"] != cutoff or receipt["validation_cycle_min"] <= cutoff:
                raise AssertionError(f"Temporal boundary crossed at {key}")
            recalculated_fit_data = _fit_fingerprint(frame.iloc[fit_rows_expected].loc[:, list(HYBRID_FEATURES)],
                                                      HYBRID_FEATURES, target[fit_rows_expected])
            if recalculated_fit_data != state["fit_data_sha256"]:
                raise AssertionError(f"Fit feature/target fingerprint mismatch at {key}")
            target_hash = hashlib.sha256(np.asarray(target[fit_rows_expected], dtype="<f8").tobytes(order="C")).hexdigest()
            if target_hash != state["target_sha256"]:
                raise AssertionError(f"Fit target digest mismatch at {key}")
            if state["validation_data_passed_to_fit"] or state["early_stopping_used"]:
                raise AssertionError(f"Validation or early stopping used during fit at {key}")
            if state["svd_used"] or state["native_categorical_features_used"] or not state["sparse_design"]:
                raise AssertionError(f"Representation differs from the preregistered sparse design at {key}")
            if state["training_design_shape"][0] != len(fit_rows_expected):
                raise AssertionError(f"Fitted design row count is not fit-only at {key}")
            if receipt["prediction_diagnostics"]["n_predictions"] != len(val_rows_expected):
                raise AssertionError(f"Prediction diagnostic row count mismatch at {key}")
            fit_receipts[key] = {
                "fit_rows": len(fit_rows_expected),
                "fit_positions_sha256": _array_hash(fit_rows_expected),
                "validation_rows": len(val_rows_expected),
                "validation_positions_sha256": _array_hash(val_rows_expected),
                "fit_feature_target_sha256_recomputed": recalculated_fit_data,
                "raw_prediction_sha256": _array_hash(candidate_raw, "<f8"),
                "passes_fit_boundary": True,
            }

    actual_views: dict[str, Any] = {}
    view_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    baseline_maps: dict[int, dict[int, int]] = {}
    for cutoff in (55, 67):
        baseline_maps[cutoff] = {int(row): i for i, row in enumerate(baseline_data[f"rows_{cutoff}"])}
    for view, (cutoff, low, high) in VIEWS.items():
        rows = np.flatnonzero((cycle >= low) & (cycle <= high)).astype(np.int64)
        if np.any(cycle[rows] <= cutoff):
            raise AssertionError(f"Fit/validation overlap in {view}")
        baseline_index = np.asarray([baseline_maps[cutoff][int(row)] for row in rows], dtype=np.int64)
        ridge25 = np.asarray(baseline_data[f"blend_ridge25_{cutoff}"][baseline_index], dtype=float)
        frame_view = frame.iloc[rows]
        if _array_hash(rows) != pilot["views"][view]["source_positions_sha256"]:
            raise AssertionError(f"View row hash mismatch for {view}")
        if not _close(_metric(target[rows], ridge25)["mae"], pilot["views"][view]["ridge25"]["mae"]):
            raise AssertionError(f"Ridge25 metric mismatch for {view}")
        actual_views[view] = {"rows": int(len(rows)), "ridge25": _metric(target[rows], ridge25)}
        view_predictions[view] = {}
        for recipe in RECIPES:
            key = f"{recipe}_{cutoff}"
            validation_rows = np.asarray(candidate_data[f"rows_{key}"], dtype=np.int64)
            row_map = {int(row): idx for idx, row in enumerate(validation_rows)}
            raw = np.asarray(candidate_data[f"raw_{key}"])[np.asarray([row_map[int(row)] for row in rows], dtype=np.int64)]
            clipped = np.clip(raw, 0.0, 100.0)
            rank50 = np.clip(0.5 * raw + 0.5 * _cycle_rank(raw, cycle[rows]), 0.0, 100.0)
            blend25 = np.clip(0.25 * rank50 + 0.75 * ridge25, 0.0, 100.0)
            estimates = {"raw": raw, "clipped": clipped, "rank50": rank50, "blend25": blend25}
            actual_views[view][recipe] = {
                name: {**_metric(target[rows], estimate), "groups": _group_metrics(frame_view, estimate)}
                for name, estimate in estimates.items()
            }
            for name, estimate in estimates.items():
                stored = pilot["views"][view][recipe][name]
                computed = actual_views[view][recipe][name]
                if stored["n"] != computed["n"] or not _close(stored["mae"], computed["mae"]):
                    raise AssertionError(f"Candidate metric mismatch for {view}/{recipe}/{name}")
                for group_name, metrics in computed["groups"].items():
                    stored_group = stored["groups"][group_name]
                    if stored_group["n"] != metrics["n"] or (metrics["n"] and not _close(stored_group["mae"], metrics["mae"])):
                        raise AssertionError(f"Subgroup metric mismatch at {view}/{recipe}/{name}/{group_name}")
            view_predictions[view][recipe] = {"rank50": rank50, "blend25": blend25}

    gate_recalculation: dict[str, Any] = {}
    for recipe in RECIPES:
        gate_recalculation[recipe] = {}
        for policy in ("rank50", "blend25"):
            primary_rows = np.concatenate([
                np.flatnonzero((cycle >= VIEWS[v][1]) & (cycle <= VIEWS[v][2])).astype(np.int64)
                for v in ("W1", "W4")
            ])
            h12_rows = np.concatenate([
                np.flatnonzero((cycle >= VIEWS[v][1]) & (cycle <= VIEWS[v][2])).astype(np.int64)
                for v in ("H12_E", "H12_L")
            ])
            def full_prediction(view_name: str) -> tuple[np.ndarray, np.ndarray]:
                cutoff = VIEWS[view_name][0]
                rows = np.flatnonzero((cycle >= VIEWS[view_name][1]) & (cycle <= VIEWS[view_name][2])).astype(np.int64)
                fit_key = f"{recipe}_{cutoff}"
                saved_rows = np.asarray(candidate_data[f"rows_{fit_key}"], dtype=np.int64)
                saved_map = {int(row): i for i, row in enumerate(saved_rows)}
                raw = np.asarray(candidate_data[f"raw_{fit_key}"])[np.asarray([saved_map[int(row)] for row in rows], dtype=np.int64)]
                base_i = np.asarray([baseline_maps[cutoff][int(row)] for row in rows], dtype=np.int64)
                baseline = np.asarray(baseline_data[f"blend_ridge25_{cutoff}"])[base_i]
                mapped = np.clip(0.5 * raw + 0.5 * _cycle_rank(raw, cycle[rows]), 0, 100)
                chosen = mapped if policy == "rank50" else np.clip(0.25 * mapped + 0.75 * baseline, 0, 100)
                return rows, chosen

            primary_parts = [full_prediction(v) for v in ("W1", "W4")]
            h12_parts = [full_prediction(v) for v in ("H12_E", "H12_L")]
            p_rows = np.concatenate([part[0] for part in primary_parts])
            p_pred = np.concatenate([part[1] for part in primary_parts])
            h_rows = np.concatenate([part[0] for part in h12_parts])
            h_pred = np.concatenate([part[1] for part in h12_parts])
            primary_delta = _metric(target[p_rows], p_pred)["mae"] - _metric(
                target[p_rows], np.concatenate([
                    np.asarray(baseline_data[f"blend_ridge25_{VIEWS[v][0]}"])[
                        np.asarray([baseline_maps[VIEWS[v][0]][int(row)] for row in part[0]], dtype=np.int64)
                    ] for v, part in zip(("W1", "W4"), primary_parts)
                ])
            )["mae"]
            h12_delta = _metric(target[h_rows], h_pred)["mae"] - _metric(
                target[h_rows], np.concatenate([
                    np.asarray(baseline_data[f"blend_ridge25_{VIEWS[v][0]}"])[
                        np.asarray([baseline_maps[VIEWS[v][0]][int(row)] for row in part[0]], dtype=np.int64)
                    ] for v, part in zip(("H12_E", "H12_L"), h12_parts)
                ])
            )["mae"]
            view_deltas = {
                view: actual_views[view][recipe][policy]["mae"] - actual_views[view]["ridge25"]["mae"]
                for view in VIEWS
            }
            passed = (primary_delta < 0 or h12_delta < 0) and all(v <= .50 for v in view_deltas.values())
            gate_recalculation[recipe][policy] = {
                "primary_delta_mae_candidate_minus_ridge25": float(primary_delta),
                "h12_delta_mae_candidate_minus_ridge25": float(h12_delta),
                "individual_view_delta_mae": {k: float(v) for k, v in view_deltas.items()},
                "pilot_continuation_gate_pass": bool(passed),
            }
            stored_gate = pilot["candidate_policy_pilot_gate"][recipe][policy]
            if stored_gate["pilot_continuation_gate_pass"] != passed:
                raise AssertionError(f"Pilot continuation decision does not recompute for {recipe}/{policy}")
            if not _close(stored_gate["primary_delta_mae_candidate_minus_ridge25"], primary_delta):
                raise AssertionError(f"Pooled primary delta does not recompute for {recipe}/{policy}")
            if not _close(stored_gate["h12_delta_mae_candidate_minus_ridge25"], h12_delta):
                raise AssertionError(f"Pooled H12 delta does not recompute for {recipe}/{policy}")

    verification = {
        "experiment_id": "optimization_v2_sparse_hybrid_regression_pilot",
        "status": "PASS",
        "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "freeze_sha256": file_sha256(FREEZE_PATH),
        "pilot_sha256": file_sha256(PILOT_PATH),
        "prediction_cache_sha256": file_sha256(PREDICTION_PATH),
        "official_train_sha256": file_sha256(train_path),
        "fit_boundary_checks": fit_receipts,
        "recomputed_views": actual_views,
        "recomputed_pilot_gates": gate_recalculation,
        "checks": {
            "four_new_fits_only": True,
            "current_ridge25_reused_row_exactly": True,
            "fit_only_preprocessing_and_target_hashes": True,
            "validation_excluded_from_model_fit": True,
            "official_test_not_loaded": True,
            "metrics_and_subgroups_recomputed_independently": True,
            "raw_clipped_rank50_and_fixed_blend_recomputed": True,
            "tails_bias_EF_stream_coverage_recomputed": True,
            "source_and_protocol_hashes_match_prefit_freeze": True,
        },
        "recommendation": "Continue with the preregistered remaining four temporal fits only if at least one fixed rank50/blend25 policy passes its pilot gate; otherwise retire the candidate family. The six-fit continuation is not included in this verification.",
    }
    _write_json_new(VERIFICATION_PATH, verification)
    baseline_data.close()
    candidate_data.close()
    print(json.dumps({"verification_path": str(VERIFICATION_PATH), "status": "PASS",
                      "gates": gate_recalculation}, indent=2))


if __name__ == "__main__":
    verify()
