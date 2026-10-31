"""Frozen R01 bridge: verified three-component control and sparse LinearSVR pilot.

The preflight-only mode validates the official Train, cached controls, declared
temporal views, and resource/fit budgets, without a freeze, output creation, or
estimator fit. This module never reads Regression Test.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    metric,
    read_bundle,
    run_pilot,
    temporal_rows,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r01 import (
    FEATURE_NAMES,
    LINEAR_SVR_PARAMETERS,
    R01ConvergenceError,
    fit_predict,
    rank50_prediction,
)
from src.models.regression_research_text import RegressionTextModel, _TFIDF_PARAMS, _text_documents
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R01.md"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/R01/freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R01"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R01"

EXPECTED_QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
EXPECTED_CARD_SHA256 = "51c960afd12dd07327ab4d93a12dffe49febc643957d86c6a21a7b4db7e72607"
EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
EXPECTED_MANIFEST_SHA256 = "dd166e92089f24978244496a9a0afabc0d06551dbdc326579a7c898a8844e19a"
EXPECTED_BASE_CACHE_SHA256 = "1227404ffa7d9f5c636c109238bfc6ee99dcbfbc128c3a07f16a2eec7ddd0516"
EXPECTED_SPARSE_CACHE_SHA256 = "eae6bea2dd6933de59ac0f63f54cc08124b59d4bcba91bbe05a7c5ec3dab9ff6"
EXPECTED_DIAGNOSTIC_CACHE_SHA256 = "283a6e91ebc904de6152103f2a1f20362d52bf82fd21172e2f8b37d48721e4a5"
EXPECTED_DIAGNOSTICS_SHA256 = "8d634debd1dbc6a3eed5250593b9c213570b4fb139b0b9a623a8d704159916df"

BASE_CACHE = ROOT / "experiments/runs/dual_reg_v1/stage1_paired_blends.npz"
SPARSE_CACHE = ROOT / "experiments/runs/optimization_v2_regression_repr/pilot_predictions.npz"
DIAGNOSTIC_CACHE = ROOT / "experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz"
DIAGNOSTICS_REPORT = ROOT / "reports/optimization_v3/regression_diagnostics/diagnostics.json"
ACCEPTED_MANIFEST = ROOT / "reports/optimization_v2/kaggle/v2_reg_sparse_lgb25.json"
FULLSIX_REPORT = ROOT / "reports/optimization_v2/regression_repr/fullsix.json"
FINDINGS_REPORT = ROOT / "reports/optimization_v2/FINDINGS.md"

PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "psutil"]
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required R01 input is missing: {label} ({path})")
    actual = file_sha256(path)
    if actual != expected:
        raise BatchIntegrityError(f"Frozen R01 {label} hash differs: {actual}")
    return actual


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    cycle_values = np.asarray(cycles, dtype=np.int64)
    if values.ndim != 1 or values.shape != cycle_values.shape or not np.isfinite(values).all():
        raise BatchIntegrityError("R01 control raw predictions/cycles are invalid")
    output = np.empty(len(values), dtype=np.float64)
    for cycle in np.unique(cycle_values):
        mask = cycle_values == cycle
        output[mask] = 0.5 * values[mask] + 0.5 * 100.0 * rankdata(
            values[mask], method="average") / int(mask.sum())
    return np.clip(output, 0.0, 100.0)


def _validate_manifest(queue: dict[str, Any]) -> dict[str, Any]:
    expected_manifest_path = ROOT / queue["baseline"]["regression"]["manifest"]
    if expected_manifest_path != ACCEPTED_MANIFEST:
        raise BatchIntegrityError("Queue points to a different accepted Regression control manifest")
    _require_hash(ACCEPTED_MANIFEST, EXPECTED_MANIFEST_SHA256, "accepted manifest")
    manifest = json.loads(ACCEPTED_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREPARED_AND_QA_PASS" or manifest.get("candidate_id") != "v2_reg_sparse_lgb25":
        raise BatchIntegrityError("Accepted Regression manifest identity/status changed")
    expected_components = [
        {"model": "lgb", "params_id": "base", "rank_weight": 0.5, "weight": 0.5625},
        {"model": "ridge", "rank_weight": 0.75, "weight": 0.1875},
        {"model": "lgbm_sparse_l1", "rank_weight": 0.5, "weight": 0.25},
    ]
    if manifest.get("recipe", {}).get("components") != expected_components:
        raise BatchIntegrityError("Accepted three-component Regression recipe differs from R01 control")
    base = queue["baseline"]["regression"]
    if (not np.isclose(float(base["primary"]), 11.559390078665364, atol=1e-12, rtol=0)
            or not np.isclose(float(base["H12"]), 14.04753258299653, atol=1e-12, rtol=0)
            or base.get("Train_sha256") != EXPECTED_TRAIN_SHA256
            or base.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256):
        raise BatchIntegrityError("Canonical R01 baseline metrics or hashes changed")
    return manifest


def _check_official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    # The canonical resolver selects only the Train path below; no Test path is opened.
    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    _require_hash(train_path, EXPECTED_TRAIN_SHA256, "official Regression Train")
    frame = pd.read_csv(train_path)
    required = {"record_id", "cycle_code", "threat_rank", *FEATURE_NAMES}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise BatchIntegrityError(f"Official Regression Train lacks required columns: {absent}")
    if len(frame) != 50_608:
        raise BatchIntegrityError(f"Unexpected Regression Train row count: {len(frame)}")
    if frame.record_id.isna().any() or frame.record_id.astype(str).str.strip().eq("").any() or not frame.record_id.astype(str).is_unique:
        raise BatchIntegrityError("Official Regression Train record_id is blank or duplicated")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80, dtype=np.int64)):
        raise BatchIntegrityError("Official Regression Train cycle coverage differs from 47–79")
    target = pd.to_numeric(frame.threat_rank, errors="raise").to_numpy(dtype=np.float64)
    if target.shape != (len(frame),) or not np.isfinite(target).all() or np.any((target < 0) | (target > 100)):
        raise BatchIntegrityError("Official Regression Train threat_rank is invalid")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R01 feature allowlist contains a protected field")
    return train_path, frame, target, cycles


def _load_controls(frame: pd.DataFrame, base: dict[str, np.ndarray],
                   sparse_cache: dict[str, np.ndarray], diagnostic: dict[str, np.ndarray]):
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        cutoff = int(cutoff)
        if cutoff not in (55, 67):
            raise BatchIntegrityError(f"R01 has no frozen control for cutoff {cutoff}")
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if not np.array_equal(score_rows, expected_rows):
            raise BatchIntegrityError(f"R01 exact score rows differ from official cycle view at {cutoff}")
        base_rows = np.asarray(base[f"rows_{cutoff}"], dtype=np.int64)
        sparse_rows = np.asarray(sparse_cache[f"rows_lgbm_sparse_l1_{cutoff}"], dtype=np.int64)
        if not np.array_equal(base_rows, expected_rows) or not np.array_equal(sparse_rows, expected_rows):
            raise BatchIntegrityError(f"R01 cache rows do not exactly match Train score rows at {cutoff}")

        lgb = np.asarray(base[f"lgb_{cutoff}"], dtype=np.float64)
        ridge = np.asarray(base[f"ridge_{cutoff}"], dtype=np.float64)
        ridge25_cached = np.asarray(base[f"blend_ridge25_{cutoff}"], dtype=np.float64)
        ridge25 = 0.75 * lgb + 0.25 * ridge
        if ridge25.shape != expected_rows.shape or not np.array_equal(ridge25_cached, ridge25):
            raise BatchIntegrityError(f"R01 cached Ridge25 does not reconstruct exactly at {cutoff}")

        sparse_raw = np.asarray(sparse_cache[f"raw_lgbm_sparse_l1_{cutoff}"], dtype=np.float64)
        if sparse_raw.shape != expected_rows.shape or not np.isfinite(sparse_raw).all():
            raise BatchIntegrityError(f"R01 sparse LGB control is invalid at {cutoff}")
        score_cycles = cycles[expected_rows]
        sparse_rank = _rank50(sparse_raw, score_cycles)
        champion = np.clip(0.75 * ridge25 + 0.25 * sparse_rank, 0.0, 100.0)
        for name, values in (("lgb_svd32", lgb), ("hybrid_ridge", ridge),
                             ("sparse_rank50", sparse_rank), ("champion", champion)):
            if values.shape != (len(expected_rows),) or not np.isfinite(values).all():
                raise BatchIntegrityError(f"R01 {name} control is invalid at cutoff {cutoff}")

        # Independent row-aligned V3 diagnostic cache is a second exact reconstruction check.
        diagnostic_rows = np.asarray(diagnostic["h12_rows"], dtype=np.int64)
        positions = {int(row): index for index, row in enumerate(diagnostic_rows)}
        try:
            diagnostic_positions = np.asarray([positions[int(row)] for row in expected_rows], dtype=np.int64)
        except KeyError as exc:
            raise BatchIntegrityError(f"R01 champion diagnostic cache omits score row {exc.args[0]}") from exc
        if not np.array_equal(np.asarray(diagnostic["h12_prediction"])[diagnostic_positions], champion):
            raise BatchIntegrityError(f"R01 three-component champion differs from diagnostic cache at {cutoff}")
        diagnostic_parts = (
            ("h12_lgb_svd32", lgb),
            ("h12_hybrid_ridge", ridge),
            ("h12_sparse_lgb_rank50", sparse_rank),
        )
        for key, values in diagnostic_parts:
            if not np.array_equal(np.asarray(diagnostic[key])[diagnostic_positions], values):
                raise BatchIntegrityError(f"R01 champion component {key} differs from diagnostic cache")
        return {
            "baseline_prediction": champion,
            "baseline_lgb_svd32": lgb,
            "baseline_hybrid_ridge": ridge,
            "baseline_sparse_raw": sparse_raw,
            "baseline_sparse_rank50": sparse_rank,
        }

    return load


def _validate_views_and_controls(frame: pd.DataFrame, target: np.ndarray,
                                 cycles: np.ndarray, views: dict[str, list[tuple[int, int, int]]],
                                 control_loader: Any, queue: dict[str, Any],
                                 diagnostic: dict[str, np.ndarray]) -> dict[str, Any]:
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R01 evaluation views differ from its frozen card: {views}")
    expected_view_rows = {"primary": int(queue["evaluation"]["STANDARD"]["pilot_primary_rows"]),
                          "H12": int(queue["evaluation"]["STANDARD"]["H12_rows"])}
    controls_by_cutoff: dict[int, dict[str, np.ndarray]] = {}
    fit_counts: dict[str, int] = {}
    for cutoff in (55, 67):
        fit_rows, _ = temporal_rows(frame, cutoff, horizon=12)
        score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if np.intersect1d(fit_rows, score_rows).size:
            raise BatchIntegrityError(f"R01 train/score rows overlap at cutoff {cutoff}")
        fit_counts[str(cutoff)] = int(len(fit_rows))
        controls_by_cutoff[cutoff] = dict(control_loader(cutoff, score_rows))

    view_metrics: dict[str, Any] = {}
    for name, segments in views.items():
        row_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            full_score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
            mask = (cycles[full_score_rows] >= low) & (cycles[full_score_rows] <= high)
            selected_rows = full_score_rows[mask]
            row_parts.append(selected_rows)
            prediction_parts.append(controls_by_cutoff[cutoff]["baseline_prediction"][mask])
        rows = np.concatenate(row_parts)
        pred = np.concatenate(prediction_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R01 view {name} contains duplicate Train rows")
        if name in expected_view_rows and len(rows) != expected_view_rows[name]:
            raise BatchIntegrityError(f"R01 {name} denominator differs: {len(rows)}")
        view_metrics[name] = {"rows": int(len(rows)), "baseline": metric("regression", target[rows], pred)["mae"]}

    # Queue baseline MAEs are accepted full-six metrics. R01 STANDARD primary
    # is only W1+W4; compare those separately, and bind accepted MAEs to the
    # independently saved full-six diagnostic rows/predictions.
    base = queue["baseline"]["regression"]
    accepted_metric_check: dict[str, Any] = {}
    for name in ("primary", "h12"):
        rows = np.asarray(diagnostic[f"{name}_rows"], dtype=np.int64)
        diagnostic_expected_rows = np.flatnonzero(np.isin(cycles, np.unique(cycles[rows])))
        if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), diagnostic_expected_rows):
            raise BatchIntegrityError(f"Accepted {name} diagnostic rows differ from exact Train view membership")
        if not np.array_equal(np.asarray(diagnostic[f"{name}_target"], dtype=np.float64), target[rows]):
            raise BatchIntegrityError(f"Accepted {name} diagnostic labels differ from official Train")
        accepted = metric("regression", target[rows],
                          np.asarray(diagnostic[f"{name}_prediction"], dtype=np.float64))["mae"]
        expected = float(base["primary" if name == "primary" else "H12"])
        if not np.isclose(accepted, expected, atol=1e-10, rtol=0):
            raise BatchIntegrityError(f"Accepted full-six {name} metric {accepted} != {expected}")
        accepted_metric_check[name] = {"rows": int(len(rows)), "mae": float(accepted)}
    return {"fit_rows_by_cutoff": fit_counts, "views": view_metrics,
            "exact_component_reconstruction": True,
            "pilot_primary_rows": expected_view_rows["primary"],
            "pilot_H12_rows": expected_view_rows["H12"],
            "accepted_fullsix_metrics": accepted_metric_check}


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray, card: dict[str, Any],
                        queue: dict[str, Any]) -> dict[str, Any]:
    preset = queue["execution_preset"]
    fit_rows = np.flatnonzero(cycles <= 67)
    documents, _ = _text_documents(frame.iloc[fit_rows]["signal_stream"])
    analyzer = TfidfVectorizer(**_TFIDF_PARAMS).build_analyzer()
    # Fit-free upper bound: count raw word unigrams+bigrams before vocabulary pruning.
    nnz_upper = int(sum(len(analyzer(document)) for document in documents))
    sparse_upper_bytes = int(nnz_upper * 16 + (len(fit_rows) + 1) * 8)
    estimate_gib = float(1.5 + 3.0 * sparse_upper_bytes / (1024 ** 3))
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    cap_gib = float(preset["max_job_RSS_GiB"])
    reserve_gib = 2.0
    minimum_free_gib = float(preset["minimum_free_RAM_GiB_before_start"])
    allowed_gib = min(cap_gib, available_gib - reserve_gib)
    if available_gib < minimum_free_gib or estimate_gib > allowed_gib:
        raise BatchIntegrityError(
            f"R01 resource gate blocked: available={available_gib:.2f}GiB, "
            f"estimated_peak={estimate_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB")
    if int(preset["concurrent_fit_jobs"]) != 1 or int(preset["threads_per_job"]) != 2:
        raise BatchIntegrityError("R01 requires one serialized fit job with two threads")
    if int(card.get("pilot_predictor_fits", -1)) != 2:
        raise BatchIntegrityError("R01 declared pilot fit count changed from two")
    if "no dense full-word" not in str(preset.get("RAM_policy", "")).casefold():
        raise BatchIntegrityError("R01 queue no longer prohibits dense full-word matrices")

    ledger = REPORT_DIR.parent / "batch_ledger.jsonl"
    reservations: dict[str, int] = {}
    if ledger.is_file():
        for line_no, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid batch fit ledger line {line_no}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key, count = str(event["budget_key"]), int(event["predictor_fits"])
                if key in reservations and reservations[key] != count:
                    raise BatchIntegrityError(f"Conflicting fit reservation in ledger for {key}")
                reservations[key] = count
    key = "R01:PILOT"
    if key in reservations and reservations[key] != int(card["pilot_predictor_fits"]):
        raise BatchIntegrityError("Existing R01 fit reservation differs from the card")
    total_after = sum(reservations.values()) + (0 if key in reservations else int(card["pilot_predictor_fits"]))
    fit_cap = int(preset["max_successful_new_predictor_fits"])
    if total_after > fit_cap:
        raise BatchIntegrityError(f"R01 fit budget blocked: reserved_after={total_after}, cap={fit_cap}")
    return {
        "fit_rows_at_largest_cutoff": int(len(fit_rows)),
        "word_ngram_nnz_upper_bound": nnz_upper,
        "sparse_design_upper_gib": float(sparse_upper_bytes / (1024 ** 3)),
        "estimated_peak_job_gib": estimate_gib,
        "available_ram_gib_observed": available_gib,
        "allowed_job_ram_gib": allowed_gib,
        "max_job_ram_gib": cap_gib,
        "minimum_free_ram_gib": minimum_free_gib,
        "reserved_fit_total_after_R01": total_after,
        "fit_budget_cap": fit_cap,
        "card_predictor_fits": int(card["pilot_predictor_fits"]),
        "concurrent_fit_jobs": 1,
        "threads": 2,
        "no_dense_full_word_matrix": True,
        "estimator_fits_performed": 0,
    }


def _input_files(train_path: Path) -> dict[str, Path]:
    return {
        "official_regression_train": train_path,
        "queue": QUEUE_PATH,
        "card": CARD_PATH,
        "accepted_v2_manifest": ACCEPTED_MANIFEST,
        "v2_findings": FINDINGS_REPORT,
        "v2_regression_fullsix": FULLSIX_REPORT,
        "v3_regression_diagnostics": DIAGNOSTICS_REPORT,
        "v2_dense_blend_cache": BASE_CACHE,
        "v2_sparse_pilot_cache": SPARSE_CACHE,
        "v3_exact_champion_cache": DIAGNOSTIC_CACHE,
    }


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1_r01.py",
        ROOT / "src/models/optimization_v2_regression.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_research_tabular.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "scripts/optimization_v3_reg_diagnostics.py",
        ROOT / "src/utils/paths.py",
    ]


def reconstruct_r01(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent rowwise reconstruction of the fixed policy and V2 champion."""
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    cycles = np.asarray(arrays["audit__score_cycles"], dtype=np.int64)
    lgb = np.asarray(arrays["control__baseline_lgb_svd32"], dtype=np.float64)
    ridge = np.asarray(arrays["control__baseline_hybrid_ridge"], dtype=np.float64)
    old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    stored_old_rank = np.asarray(arrays["control__baseline_sparse_rank50"], dtype=np.float64)
    lengths = {len(x) for x in (raw, cycles, lgb, ridge, old_sparse, stored_old_rank)}
    if len(lengths) != 1:
        raise BatchIntegrityError("R01 saved route inputs have inconsistent row counts")
    old_sparse_rank = rank50_prediction(old_sparse, cycles)
    if not np.array_equal(stored_old_rank, old_sparse_rank):
        raise BatchIntegrityError("R01 saved champion sparse-rank component does not reconstruct")
    ridge25 = 0.75 * lgb + 0.25 * ridge
    baseline = np.clip(0.75 * ridge25 + 0.25 * old_sparse_rank, 0.0, 100.0)
    challenger = np.clip(0.75 * baseline + 0.25 * rank50_prediction(raw, cycles), 0.0, 100.0)
    return challenger, baseline


def _view_bundle_data(frame: pd.DataFrame, target: np.ndarray, views: dict[str, list[tuple[int, int, int]]],
                      freeze_signature: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cycle_values = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    row_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    for cutoff, low, high in views[name]:
        path = ARTIFACT_DIR / f"R01_fit{cutoff}.npz"
        sidecar_path = path.with_suffix(".json")
        if not sidecar_path.is_file():
            raise BatchIntegrityError(f"R01 saved fit bundle is missing: {path}")
        identity = json.loads(sidecar_path.read_text(encoding="utf-8")).get("identity", {})
        if identity.get("freeze_signature") != freeze_signature:
            raise BatchIntegrityError("R01 fit bundle belongs to a different freeze")
        arrays, _ = read_bundle(path, identity)
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        mask = (cycle_values[rows] >= low) & (cycle_values[rows] <= high)
        selected = rows[mask]
        row_parts.append(selected)
        y_parts.append(target[selected])
        pred_parts.append(np.asarray(arrays["prediction"], dtype=np.float64)[mask])
        base_parts.append(np.asarray(arrays["baseline_prediction"], dtype=np.float64)[mask])
    rows = np.concatenate(row_parts)
    if len(np.unique(rows)) != len(rows):
        raise BatchIntegrityError(f"R01 persisted view {name} has duplicate rows")
    return rows, np.concatenate(y_parts), np.concatenate(pred_parts), np.concatenate(base_parts)


def _subgroup_diagnostics(frame: pd.DataFrame, target: np.ndarray, rows: np.ndarray,
                          y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    if not (rows.shape == y.shape == pred.shape == baseline.shape):
        raise BatchIntegrityError("R01 subgroup arrays are not aligned")
    stream_missing = frame.iloc[rows]["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    local_target = target[rows]
    groups = {
        "stream_missing": {"missing": stream_missing, "present": ~stream_missing},
        "target_tail": {"low_le5": local_target <= 5.0, "high_ge95": local_target >= 95.0},
    }
    result: dict[str, Any] = {"support_minimum": 100, "groups": {}}
    for field, masks in groups.items():
        result["groups"][field] = {}
        for label, mask in masks.items():
            n = int(mask.sum())
            supported = n >= 100
            candidate_mae = float(np.mean(np.abs(pred[mask] - y[mask]))) if n else None
            baseline_mae = float(np.mean(np.abs(baseline[mask] - y[mask]))) if n else None
            result["groups"][field][label] = {
                "n": n, "supported": supported,
                "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                "candidate_minus_baseline_mae": (candidate_mae - baseline_mae)
                if candidate_mae is not None and baseline_mae is not None else None,
                "tail_definition_uses_observed_target_after_prediction": field == "target_tail",
            }
    return result


def _write_or_match(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise BatchIntegrityError(f"Existing R01 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _recipe(card: dict[str, Any], views: dict[str, list[tuple[int, int, int]]]) -> dict[str, Any]:
    from sklearn.svm import LinearSVR

    return {
        "card_id": "R01",
        "status": "FROZEN_RECIPE_BEFORE_PILOT",
        "card_sha256": EXPECTED_CARD_SHA256,
        "accepted_control": {
            "manifest": str(ACCEPTED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "route": ".5625 rank50 LightGBM SVD32 + .1875 rank75 Hybrid Ridge + .25 rank50 sparse LGB",
            "pilot_control_is_exact_v2_three_component_champion": True,
        },
        "features": list(FEATURE_NAMES),
        "preprocessing": {
            "structured": "RegressionTextModel._structured_fit/_structured_transform; fold-fit imputers, scaling and rare-category OHE",
            "categorical_min_frequency": 30,
            "categorical_max_categories": 254,
            "text": {
                key: (list(value) if isinstance(value, tuple)
                      else value.__name__ if isinstance(value, type)
                      else value.item() if isinstance(value, np.generic)
                      else value)
                for key, value in _TFIDF_PARAMS.items()
            },
            "text_and_structured_fit_scope": "fit rows of each cutoff only",
            "missing_signal_stream_indicator": True,
            "sparse_design": True,
            "svd": False,
        },
        "estimator": {
            "class": "sklearn.svm.LinearSVR",
            "parameters_explicit": dict(LINEAR_SVR_PARAMETERS),
            "parameters_effective": LinearSVR(**LINEAR_SVR_PARAMETERS).get_params(deep=False),
            "target_transform": "z=(threat_rank-50)/50",
            "inverse": "prediction=50*z+50",
            "convergence_rule": "record ConvergenceWarning and n_iter; halt before scoring if warning or max_iter reached; no post-score iteration rescue",
            "regularization_equivalence_claim": "not claimed equivalent to historical SGDRegressor",
        },
        "policy": ".75 current three-component champion + .25 within-cycle rank50(new target-scale predictor), clipped to [0,100]",
        "evaluation_views": views,
        "support": {"pilot_primary_rows": 9064, "H12_rows": 38588, "subgroup_n_min": 100},
        "pilot_predictor_fit_count": int(card["pilot_predictor_fits"]),
        "metric": "MAE, positive gain = baseline MAE - candidate MAE",
        "gates": {
            "pilot_primary_MAE_gain_min": 0.03,
            "pilot_H12_MAE_harm_max": 0.05,
            "individual_pilot_view_MAE_harm_max": 0.5,
            "supported_tail_or_missing_MAE_harm_max": 0.25,
        },
        "test_data_used": False,
    }


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    if card.get("id") != "R01" or card.get("task") != "regression":
        raise BatchIntegrityError("R01 runner received a different queue card")
    _require_hash(QUEUE_PATH, EXPECTED_QUEUE_SHA256, "canonical queue")
    if load_queue(QUEUE_PATH) != queue:
        raise BatchIntegrityError("Dispatcher queue differs from the frozen R01 queue")
    if (ROOT / card["card_path"]) != CARD_PATH:
        raise BatchIntegrityError("Canonical queue points to an unexpected R01 card path")
    _require_hash(CARD_PATH, EXPECTED_CARD_SHA256, "R01 card")
    _require_hash(BASE_CACHE, EXPECTED_BASE_CACHE_SHA256, "V2 paired-blend cache")
    _require_hash(SPARSE_CACHE, EXPECTED_SPARSE_CACHE_SHA256, "V2 sparse pilot cache")
    _require_hash(DIAGNOSTIC_CACHE, EXPECTED_DIAGNOSTIC_CACHE_SHA256, "V3 champion diagnostic cache")
    _require_hash(DIAGNOSTICS_REPORT, EXPECTED_DIAGNOSTICS_SHA256, "V3 regression diagnostics report")
    for path in (FINDINGS_REPORT, FULLSIX_REPORT):
        if not path.is_file():
            raise FileNotFoundError(f"R01 evidence report missing: {path}")
    _validate_manifest(queue)
    train_path, frame, target, cycles = _check_official_train()

    views = evaluation_views_for_card(queue, "R01")
    base_cache = _read_npz(BASE_CACHE)
    sparse_cache = _read_npz(SPARSE_CACHE)
    diagnostic_cache = _read_npz(DIAGNOSTIC_CACHE)
    control_loader = _load_controls(frame, base_cache, sparse_cache, diagnostic_cache)
    control_check = _validate_views_and_controls(
        frame, target, cycles, views, control_loader, queue, diagnostic_cache)
    resources = _resource_preflight(frame, cycles, card, queue)

    inputs = _input_files(train_path)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=inputs,
        source_files=_source_files(), evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "accepted_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "exact_control_cache_sha256": {
            "paired_blends": EXPECTED_BASE_CACHE_SHA256,
            "sparse_pilot": EXPECTED_SPARSE_CACHE_SHA256,
            "independent_champion": EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        },
        "control_reconstruction": control_check,
        "resource_preflight_recipe": {
            "estimated_peak_job_gib": resources["estimated_peak_job_gib"],
            "cap_gib": resources["max_job_ram_gib"],
            "minimum_free_ram_gib": resources["minimum_free_ram_gib"],
            "reserve_gib": 2.0,
            "estimate_method": "fit-only raw word n-gram upper bound; sparse bytes x3 plus 1.5GiB fixed",
            "threads": 2,
            "concurrent_fit_jobs": 1,
        },
        "python_version": platform.python_version(),
        "test_data_used": False,
        "validation_labels_passed_to_adapter": False,
    })

    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT",
            "card_id": "R01",
            "train_sha256": EXPECTED_TRAIN_SHA256,
            "control_check": control_check,
            "resource_preflight": resources,
            "source_file_count": len(_source_files()),
            "input_file_count": len(inputs),
            "evaluation_views": views,
            "freeze_identity_sha256": json_sha256(identity),
            "official_test_read": False,
            "model_fit_performed": False,
            "freeze_or_output_written": False,
        }

    freeze_signature = json_sha256(identity)
    failure_path = REPORT_DIR / "failure.json"
    gate_path = REPORT_DIR / "gate.json"
    event_path = REPORT_DIR / "attempts.jsonl"
    if failure_path.is_file():
        prior_failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if prior_failure.get("card_id") != "R01" or prior_failure.get("freeze_signature") != freeze_signature:
            raise BatchIntegrityError("Existing R01 terminal failure belongs to a different frozen identity")
        if not gate_path.is_file():
            raise BatchIntegrityError("R01 terminal failure exists without its gate receipt; refusing retry")
        prior_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if prior_gate.get("status") != "RETIRE" or prior_gate.get("freeze_signature") != freeze_signature:
            raise BatchIntegrityError("Existing R01 failure gate does not match its failure receipt")
        return {"failure": prior_failure, "gate": prior_gate, "status": "RETIRE"}
    if event_path.is_file():
        prior_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(event.get("card_id") == "R01" and event.get("event") == "FIT_FAILED"
               and event.get("error_type") == "R01ConvergenceError" for event in prior_events):
            raise BatchIntegrityError("Prior R01 convergence failure lacks terminal receipt; refusing a retry or rescue")

    recipe = _recipe(card, views)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_or_match(REPORT_DIR / "recipe.json", recipe)
    try:
        report = run_pilot(
            frame=frame, target=target, card=card, feature_names=FEATURE_NAMES,
            adapter=fit_predict, control_loader=control_loader,
            freeze_identity=identity, freeze_path=FREEZE_PATH,
            artifact_dir=ARTIFACT_DIR, event_path=event_path,
            evaluation_views=views, threads=2, report_path=REPORT_DIR / "pilot.json",
            budget_ledger_path=REPORT_DIR.parent / "batch_ledger.jsonl",
            max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        )
    except R01ConvergenceError as exc:
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        matching = [event for event in events if event.get("card_id") == "R01"
                    and event.get("event") == "FIT_FAILED"
                    and event.get("error_type") == "R01ConvergenceError"]
        if not matching:
            raise BatchIntegrityError("R01 convergence error was not preserved by the generic FIT_FAILED receipt") from exc
        engine_failure = matching[-1]
        expected_prefix = (
            f"R01_CONVERGENCE_FAILED n_iter={exc.n_iter} max_iter={exc.max_iter} "
            f"converged={exc.converged} warning_count={len(exc.warning_messages)};"
        )
        if not str(engine_failure.get("error", "")).startswith(expected_prefix):
            raise BatchIntegrityError("Generic FIT_FAILED receipt lost R01 convergence diagnostics") from exc
        if exc.cutoff not in (55, 67) or exc.fit_rows is None:
            raise BatchIntegrityError("R01 convergence error lacks temporal fit context") from exc
        failed_bundle = ARTIFACT_DIR / f"R01_fit{exc.cutoff}.npz"
        if failed_bundle.exists() or failed_bundle.with_suffix(".json").exists():
            raise BatchIntegrityError("R01 failed cutoff unexpectedly has a saved prediction bundle") from exc
        successful_events = [event for event in events if event.get("card_id") == "R01"
                             and event.get("event") == "FIT_SAVED"]
        started_events = [event for event in events if event.get("card_id") == "R01"
                          and event.get("event") == "FIT_STARTED"]
        failure = {
            "status": "RETIRE",
            "card_id": "R01",
            "reason": "LinearSVR did not converge within the frozen iteration budget",
            "freeze_signature": freeze_signature,
            "freeze_identity_sha256": json_sha256(identity),
            "cutoff": int(exc.cutoff),
            "fit_rows": int(exc.fit_rows),
            "n_iter": int(exc.n_iter),
            "max_iter": int(exc.max_iter),
            "converged": bool(exc.converged),
            "convergence_warning_messages": list(exc.warning_messages),
            "reserved_predictor_fits": int(card["pilot_predictor_fits"]),
            "new_fit_started_events": int(len(started_events)),
            "successful_fit_bundles_before_failure": int(len(successful_events)),
            "failed_cutoff_prediction_bundle_written": False,
            "failed_cutoff_scored": False,
            "pilot_aggregate_metrics": "NOT_COMPUTED",
            "official_test_read": False,
            "validation_labels_passed_to_model": False,
            "retry_or_iteration_rescue_performed": False,
            "engine_failure_receipt": {
                "path": str(event_path.relative_to(ROOT)),
                "event": engine_failure,
            },
        }
        failure_gate = {
            "card_id": "R01",
            "profile": card["evaluation"],
            "status": "RETIRE",
            "terminal": True,
            "freeze_signature": freeze_signature,
            "decision_basis": "Frozen convergence stop rule; no score or aggregate metrics were produced for this pilot.",
            "checks": {
                "converged_within_iteration_budget": False,
                "failed_cutoff_scored": False,
                "failed_cutoff_prediction_written": False,
                "official_test_used": False,
                "retry_or_rescue_performed": False,
            },
            "metrics": "NOT_COMPUTED",
            "failure_receipt": str(failure_path.relative_to(ROOT)),
            "engine_failure_receipt": str(event_path.relative_to(ROOT)),
        }
        _write_or_match(failure_path, failure)
        _write_or_match(gate_path, failure_gate)
        return {"failure": failure, "gate": failure_gate, "status": "RETIRE"}
    verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=ARTIFACT_DIR,
        freeze_signature=report["freeze_signature"], evaluation_views=views,
        prediction_reconstructors={"R01": reconstruct_r01},
    )

    view_detail: dict[str, Any] = {}
    subgroup_failures: list[dict[str, Any]] = []
    for name in views:
        rows, y, pred, baseline = _view_bundle_data(frame, target, views, report["freeze_signature"], name)
        subgroups = _subgroup_diagnostics(frame, target, rows, y, pred, baseline)
        view_detail[name] = {"rows": int(len(rows)), "subgroups": subgroups}
        for field in ("stream_missing", "target_tail"):
            for label, record in subgroups["groups"][field].items():
                if record["supported"] and float(record["candidate_minus_baseline_mae"]) > 0.25:
                    subgroup_failures.append({"view": name, "field": field, "group": label, **record})

    gates = queue["evaluation"]["regression_gate"]
    checks = {
        "pilot_primary_MAE_gain_min": report["views"]["primary"]["gain_positive_is_better"] >= float(gates["pilot_primary_MAE_gain_min"]),
        "pilot_H12_MAE_harm_max": report["views"]["H12"]["gain_positive_is_better"] >= -float(gates["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_max": report["views"]["W1"]["gain_positive_is_better"] >= -float(gates["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_max": report["views"]["W4"]["gain_positive_is_better"] >= -float(gates["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_harm_max": len(subgroup_failures) == 0,
        "independent_metric_and_prediction_reconstruction": verification.get("status") == "PASS",
    }
    gate = {
        "card_id": "R01", "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": gates, "supported_tail_or_missing_failures": subgroup_failures,
        "view_subgroups": view_detail,
        "decision_basis": "Predeclared temporal pilot metrics and support-passing subgroup guards; no Test or Kaggle result used.",
    }
    _write_or_match(REPORT_DIR / "verification.json", verification)
    _write_or_match(REPORT_DIR / "gate.json", gate)
    summary = (
        "# R01 Pilot Summary\n\n"
        f"- Status: **{gate['status']}**\n"
        f"- Primary MAE gain (positive is better): {report['views']['primary']['gain_positive_is_better']:.6f} "
        f"({report['views']['primary']['rows']:,} rows).\n"
        f"- H12 MAE gain (positive is better): {report['views']['H12']['gain_positive_is_better']:.6f} "
        f"({report['views']['H12']['rows']:,} rows).\n"
        f"- W1/W4 gains: {report['views']['W1']['gain_positive_is_better']:.6f} / "
        f"{report['views']['W4']['gain_positive_is_better']:.6f}.\n"
        "- The pilot is temporal development evidence with reused Train outcomes, not untouched holdout evidence.\n"
        "- Test data was not read.\n"
    )
    summary_path = REPORT_DIR / "summary.md"
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing R01 summary differs; refusing overwrite")
    else:
        with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary)
    return {"pilot": report, "verification": verification, "gate": gate,
            "control_check": control_check, "resource_preflight": resources}


__all__ = ["reconstruct_r01", "run_card"]
