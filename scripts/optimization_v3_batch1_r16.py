"""R16 read-only preflight, frozen pilot, and independent verifier.

R16 changes only the fit weighting of the SVD32 LightGBM component. All
temporal rows remain in each fit; preprocessing stays fit-only and unweighted.
The runner never opens Regression Test or SampleSubmission.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import optimization_v3_batch1_r01 as r01
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    package_versions,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r16 import (
    FEATURE_NAMES,
    fit_predict as r16_fit_predict,
)
from src.models.regression_research_text import HYBRID_FEATURES
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
CARD_ID = "R16"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R16.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R16"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R16"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R16"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
PILOT_REPORT_PATH = ROOT / "reports/optimization_v2/regression_repr/pilot.json"
PILOT_VERIFICATION_PATH = ROOT / "reports/optimization_v2/regression_repr/verification.json"
HISTORICAL_FULLSIX_FREEZE_PATH = ROOT / "experiments/optimization_v2/regression_repr/fullsix_freeze.json"

TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
CARD_SHA256 = "459708699043ac6bf1ca95364242e27d321988a0f32bcdac41d7613fd6b562ed"
MANIFEST_SHA256 = r01.EXPECTED_MANIFEST_SHA256
MIN_GROUP_SUPPORT = 100
MAX_SUCCESSFUL_FITS = 180
THREADS = 2
JOB_CAP_BYTES = 6 * 1024 ** 3
FREE_RAM_RESERVE_BYTES = 2 * 1024 ** 3
EXPECTED_FIT_ROWS = {55: 12_020, 67: 29_705}
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064,
                      "H12_E": 17_685, "H12_L": 20_903, "H12": 38_588}
EXPECTED_CONTROL_METRICS = {
    "primary_mae": 11.559390078665364,
    "H12_mae": 14.04753258299653,
}
EXPECTED_PILOT_CONTROL_METRICS = {
    "primary_mae": 13.464395831218011,
    "H12_mae": 14.04753258299653,
}
EXPECTED_GATE = {
    "pilot_primary_MAE_gain_min": 0.03,
    "pilot_H12_MAE_harm_max": 0.05,
    "individual_pilot_view_MAE_harm_max": 0.5,
    "supported_group_n_min": 100,
    "supported_tail_or_missing_MAE_harm_max": 0.25,
}
EXPECTED_CARD = {
    "id": "R16",
    "task": "regression",
    "name": "All-past exponentially weighted component",
    "family": "recency",
    "priority": "P1",
    "change": "Refit only SVD32 LGB using fit-only sample_weight=2^(-(cutoff-cycle)/8), normalize mean1. Keep all past rows/transforms and original hyperparameters. Replace only weight0.5625 incumbent component using inherited rank50.",
    "policy": "R_SVD_REPLACE",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "card_path": "reports/optimization_v3/research_backlog/cards/R16.md",
}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}

INPUTS = {
    "official_regression_train": task_files("regression")["train"],
    "canonical_queue": QUEUE_PATH,
    "canonical_card": CARD_PATH,
    "accepted_regression_manifest": r01.ACCEPTED_MANIFEST,
    "paired_lgb_ridge_prediction_cache": r01.BASE_CACHE,
    "sparse_lgb_pilot_cache": r01.SPARSE_CACHE,
    "sparse_lgb_fullsix_cache": ROOT / "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz",
    "incumbent_component_diagnostic_cache": r01.DIAGNOSTIC_CACHE,
    "regression_diagnostics_report": r01.DIAGNOSTICS_REPORT,
    "accepted_fullsix_freeze": r01.ROOT / "experiments/optimization_v2/regression_repr/fullsix_freeze.json",
    "accepted_fullsix_report": r01.FULLSIX_REPORT,
    "accepted_pilot_report": r01.ROOT / "reports/optimization_v2/regression_repr/pilot.json",
    "accepted_pilot_verification": r01.ROOT / "reports/optimization_v2/regression_repr/verification.json",
    "regression_round2_findings": r01.FINDINGS_REPORT,
}
SOURCES = [
    Path(__file__),
    ROOT / "src/models/optimization_v3_batch1_r16.py",
    ROOT / "tests/test_optimization_v3_batch1_r16.py",
    ROOT / "src/models/optimization_v3_batch1.py",
    ROOT / "scripts/optimization_v3_batch1_verify.py",
    ROOT / "scripts/optimization_v3_batch1_r01.py",
    ROOT / "src/models/optimization_v3_batch1_r01.py",
    ROOT / "scripts/optimization_v3_reg_diagnostics.py",
    ROOT / "src/models/regression_research_lgb_text.py",
    ROOT / "src/models/regression_research_text.py",
    ROOT / "src/models/regression_research_tabular.py",
    ROOT / "src/models/regression_round2_models.py",
    ROOT / "src/models/round1_models.py",
    ROOT / "src/models/classification_models.py",
    ROOT / "src/utils/paths.py",
]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, payload: Mapping[str, Any]) -> None:
    cleaned = _json_clean(dict(payload))
    if path.is_file():
        if _read_json(path) != cleaned:
            raise BatchIntegrityError(f"Existing R16 output differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, cleaned)


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required R16 input is missing: {label} ({path})")
    observed = file_sha256(path)
    if observed != expected:
        raise BatchIntegrityError(f"Frozen R16 {label} hash differs: {observed}")
    return observed


def _read_budget_ledger() -> dict[str, Any]:
    total = 0
    own: int | None = None
    if not LEDGER_PATH.is_file():
        return {"status": "PASS", "reserved_total": 0, "next_total": 2,
                "existing_r16_reservation": None, "cap": MAX_SUCCESSFUL_FITS}
    try:
        for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "BUDGET_RESERVED":
                count = int(event["predictor_fits"])
                if count < 0:
                    raise ValueError("negative predictor-fit count")
                total += count
                if event.get("budget_key") == "R16:PILOT":
                    if own is not None and own != count:
                        raise ValueError("conflicting R16 reservations")
                    own = count
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED_LEDGER", "error": str(exc), "path": str(LEDGER_PATH)}
    next_total = total if own == 2 else total + 2
    status = "PASS" if own in (None, 2) and next_total <= MAX_SUCCESSFUL_FITS else "BLOCKED_BUDGET"
    return {"status": status, "path": str(LEDGER_PATH), "reserved_total": total,
            "next_total": next_total, "existing_r16_reservation": own,
            "cap": MAX_SUCCESSFUL_FITS}


def _validate_card_and_queue(queue: Mapping[str, Any], card: Mapping[str, Any]) -> None:
    _require_hash(QUEUE_PATH, QUEUE_SHA256, "canonical queue")
    _require_hash(CARD_PATH, CARD_SHA256, "R16 card")
    if load_queue(QUEUE_PATH) != dict(queue):
        raise BatchIntegrityError("Loaded R16 queue differs from canonical queue bytes")
    canonical = next((item for item in queue["cards"] if item.get("id") == CARD_ID), None)
    if canonical is None or canonical != dict(card):
        raise BatchIntegrityError("R16 card object differs from its canonical queue entry")
    for key, expected in EXPECTED_CARD.items():
        if card.get(key) != expected:
            raise BatchIntegrityError(f"R16 queue changed in field {key}")
    if int(card.get("pilot_predictor_fits", -1)) != 2:
        raise BatchIntegrityError("R16 must reserve exactly two pilot predictor fits")
    if tuple(FEATURE_NAMES) != tuple(HYBRID_FEATURES):
        raise BatchIntegrityError("R16 feature allowlist differs from inherited SVD32 model")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise BatchIntegrityError("R16 feature allowlist contains an identity or target field")
    baseline = queue["baseline"]["regression"]
    if (baseline.get("Train_sha256") != TRAIN_SHA256
            or baseline.get("manifest_sha256") != MANIFEST_SHA256
            or float(baseline.get("primary", float("nan"))) != EXPECTED_CONTROL_METRICS["primary_mae"]
            or float(baseline.get("H12", float("nan"))) != EXPECTED_CONTROL_METRICS["H12_mae"]):
        raise BatchIntegrityError("R16 frozen incumbent identity or control metrics changed")
    gate = queue["evaluation"]["regression_gate"]
    for key, expected in EXPECTED_GATE.items():
        if float(gate.get(key, float("nan"))) != float(expected):
            raise BatchIntegrityError(f"R16 regression gate changed at {key}")


def _validate_input_lineage() -> dict[str, Any]:
    expected_hashes = {
        "official_regression_train": TRAIN_SHA256,
        "canonical_queue": QUEUE_SHA256,
        "canonical_card": CARD_SHA256,
        "accepted_regression_manifest": MANIFEST_SHA256,
        "paired_lgb_ridge_prediction_cache": r01.EXPECTED_BASE_CACHE_SHA256,
        "sparse_lgb_pilot_cache": r01.EXPECTED_SPARSE_CACHE_SHA256,
        "incumbent_component_diagnostic_cache": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        "regression_diagnostics_report": r01.EXPECTED_DIAGNOSTICS_SHA256,
        "accepted_fullsix_freeze": "bb38f5fb7df09f0642ad18f064373f2b7051ed64e8b5d260ccb2d31b84ef909c",
        "accepted_fullsix_report": "a9686473998722b8f86a0f6a4489b4e3a75a6b6ba20c661ac564e6a5e61aea48",
    }
    hashes: dict[str, str] = {}
    for name, path in sorted(INPUTS.items()):
        if not Path(path).is_file():
            raise FileNotFoundError(f"R16 required input is missing: {name} ({path})")
        hashes[name] = file_sha256(path)
    for name, expected in expected_hashes.items():
        if hashes.get(name) != expected:
            raise BatchIntegrityError(f"R16 frozen input hash differs for {name}: {hashes.get(name)}")
    historical_freeze = _read_json(HISTORICAL_FULLSIX_FREEZE_PATH)
    fullsix_report = _read_json(r01.FULLSIX_REPORT)
    historical_freeze_sha = file_sha256(HISTORICAL_FULLSIX_FREEZE_PATH)
    if (fullsix_report.get("continuation_freeze_sha256") != historical_freeze_sha
            or fullsix_report.get("all_six_primary_windows_and_both_H12_views_completed") is not True
            or fullsix_report.get("control_fit_count") != 0
            or fullsix_report.get("new_prediction_cache_path")
            != "experiments/runs/optimization_v2_regression_repr/fullsix_predictions.npz"
            or fullsix_report.get("new_prediction_cache_sha256")
            != hashes["sparse_lgb_fullsix_cache"]):
        raise BatchIntegrityError("R16 full-six cache is not bound by the accepted regression report")
    _require_hash(PILOT_REPORT_PATH,
                  historical_freeze["existing_pilot"]["report_sha256"],
                  "accepted sparse pilot report")
    _require_hash(PILOT_VERIFICATION_PATH,
                  historical_freeze["existing_pilot"]["verification_sha256"],
                  "accepted sparse pilot verification")
    return hashes


def _memory_estimate(fit_rows: int, available_bytes: int | None = None) -> dict[str, Any]:
    # Conservative dense SVD32 design bound. Structured OHE is capped at 807
    # columns by the inherited preprocessing contract; 32 SVD columns and one
    # missing-stream flag are added. Five design copies and 1 GiB overhead
    # deliberately overstate peak use; this remains an estimate, not a measure.
    structured_width = 807
    width = structured_width + 32 + 1
    design_bytes = int(fit_rows) * width * 8
    histogram_bytes = width * 255 * 16 * THREADS * 2
    overhead_bytes = 1024 ** 3
    estimated = 5 * design_bytes + histogram_bytes + overhead_bytes
    if available_bytes is None:
        try:
            import psutil
            available_bytes = int(psutil.virtual_memory().available)
        except Exception:
            available_bytes = None
    allowed = None if available_bytes is None else min(
        JOB_CAP_BYTES, max(0, int(available_bytes) - FREE_RAM_RESERVE_BYTES))
    status = "UNKNOWN_RESOURCE" if allowed is None else (
        "PASS" if estimated <= allowed else "BLOCKED_RESOURCE")
    return {"status": status, "fit_rows": int(fit_rows),
            "feature_width_upper_bound": width,
            "design_bytes_upper_bound": design_bytes,
            "estimated_peak_bytes": estimated,
            "estimated_peak_is_observed": False,
            "available_physical_bytes": available_bytes,
            "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES,
            "job_cap_bytes": JOB_CAP_BYTES, "allowed_bytes": allowed,
            "threads": THREADS, "concurrent_training_jobs": 1,
            "dense_svd_design": True}


def _load_control_arrays(frame: pd.DataFrame, cycles: np.ndarray,
                         views: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    base = r01._read_npz(r01.BASE_CACHE)
    sparse = r01._read_npz(r01.SPARSE_CACHE)
    diagnostic = r01._read_npz(r01.DIAGNOSTIC_CACHE)
    control_loader = r01._load_controls(frame, base, sparse, diagnostic)
    controls: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        saved = control_loader(cutoff, rows)
        rows = np.asarray(rows, dtype=np.int64)
        if not np.array_equal(rows, np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))):
            raise BatchIntegrityError(f"R16 control row positions differ at cutoff {cutoff}")
        controls[cutoff] = {
            "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            "old_lgb_svd32": np.asarray(saved["baseline_lgb_svd32"], dtype=np.float64),
            "ridge_prediction": np.asarray(saved["baseline_hybrid_ridge"], dtype=np.float64),
            "sparse_rank50": np.asarray(saved["baseline_sparse_rank50"], dtype=np.float64),
            "baseline_prediction": np.asarray(saved["baseline_prediction"], dtype=np.float64),
        }
        if any(value.shape != rows.shape for key, value in controls[cutoff].items()
               if key not in {"score_record_ids"}) or controls[cutoff]["score_record_ids"].shape != rows.shape:
            raise BatchIntegrityError(f"R16 control arrays are misaligned at cutoff {cutoff}")
        rebuilt = np.clip(
            0.5625 * controls[cutoff]["old_lgb_svd32"]
            + 0.1875 * controls[cutoff]["ridge_prediction"]
            + 0.25 * controls[cutoff]["sparse_rank50"], 0.0, 100.0)
        if not np.allclose(rebuilt, controls[cutoff]["baseline_prediction"], rtol=0, atol=1e-10):
            raise BatchIntegrityError(f"R16 full incumbent reconstruction differs at cutoff {cutoff}")
        if not np.array_equal(controls[cutoff]["score_record_ids"],
                              frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str)):
            raise BatchIntegrityError(f"R16 frozen control record IDs differ at cutoff {cutoff}")
    for name, segments in views.items():
        joined = np.concatenate([
            np.flatnonzero((cycles >= low) & (cycles <= high))
            for _origin, low, high in segments
        ])
        if len(joined) != EXPECTED_VIEW_ROWS[name] or len(np.unique(joined)) != len(joined):
            raise BatchIntegrityError(f"R16 view {name} population differs from frozen denominator")
    return controls


def _baseline_metrics(context: Mapping[str, Any], target: np.ndarray) -> dict[str, Any]:
    cycles = context["cycles"]
    views: dict[str, dict[str, Any]] = {}
    for name, segments in context["views"].items():
        rows: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for cutoff, low, high in segments:
            control = context["controls"][cutoff]
            selected = control["score_rows"]
            mask = (cycles[selected] >= low) & (cycles[selected] <= high)
            rows.append(selected[mask])
            predictions.append(control["baseline_prediction"][mask])
        joined_rows = np.concatenate(rows)
        pred = np.concatenate(predictions)
        views[name] = {"rows": int(len(joined_rows)),
                       "mae": float(np.mean(np.abs(target[joined_rows] - pred)))}
    if (views["primary"]["rows"] != EXPECTED_VIEW_ROWS["primary"]
            or views["H12"]["rows"] != EXPECTED_VIEW_ROWS["H12"]
            or abs(views["primary"]["mae"] - EXPECTED_PILOT_CONTROL_METRICS["primary_mae"]) > 1e-10
            or abs(views["H12"]["mae"] - EXPECTED_PILOT_CONTROL_METRICS["H12_mae"]) > 1e-10):
        raise BatchIntegrityError("R16 row-matched pilot control MAE differs from frozen expectations")
    return views


def _recipe_payload(card: Mapping[str, Any], views: Mapping[str, Any]) -> dict[str, Any]:
    from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS
    from src.models.regression_research_text import _SVD_PARAMS, _TFIDF_PARAMS

    return _json_clean({
        "card_id": CARD_ID,
        "queue_recipe": dict(card),
        "feature_allowlist": list(FEATURE_NAMES),
        "feature_role": {"record_id": "identity_only", "threat_rank": "target_only"},
        "preprocessing": {
            "inherited_model": "RegressionResearchLGBText(params_id=base)",
            "structured": "_SelectedPreprocessor fit on all rows <= cutoff; no score rows passed to fit",
            "text": {str(key): (list(value) if isinstance(value, tuple)
                                  else value.__name__ if isinstance(value, type)
                                  else value.item() if isinstance(value, np.generic)
                                  else value)
                     for key, value in _TFIDF_PARAMS.items()},
            "tfidf_fit_scope": "all fit rows at each cutoff only",
            "svd": {**_SVD_PARAMS, "n_components_requested": 32},
            "svd_fit_scope": "all fit rows at each cutoff only",
            "missing_stream_indicator": True,
            "preprocessing_sample_weighted": False,
        },
        "estimator": {
            "class": "lightgbm.LGBMRegressor",
            "objective": "regression_l1",
            "parameters": {**COMMON_PARAMETERS, **PARAMETER_SETS["base"]},
            "early_stopping": False,
            "sample_weight": {
                "formula": "2**(-(cutoff-cycle_code)/8)",
                "scope": "fit rows only; all rows with cycle_code <= cutoff retained",
                "normalization": "divide by arithmetic mean so mean weight equals 1",
                "affects": "LightGBM estimator fit only",
            },
        },
        "prediction_policy": {
            "candidate": "clip(.5625*rank50(new SVD32 LGB)+.1875*frozen mapped Ridge+.25*frozen sparse rank50,0,100)",
            "baseline": "clip(.5625*frozen mapped SVD32 LGB+.1875*frozen mapped Ridge+.25*frozen sparse rank50,0,100)",
            "rank50": "clip(.5*raw + .5*(100*average_rank(raw)/n_cycle),0,100)",
            "ties": "average",
            "ranking_population": "all score rows in each cycle supplied by the cutoff; pilot cutoffs score complete 12-cycle horizons",
        },
        "evaluation": {
            "profile": "STANDARD",
            "views": {name: [list(segment) for segment in segments] for name, segments in views.items()},
            "cutoffs": [55, 67],
            "fit_rows": EXPECTED_FIT_ROWS,
            "predictor_fits": 2,
            "denominators": EXPECTED_VIEW_ROWS,
            "minimum_group_support": MIN_GROUP_SUPPORT,
        },
        "accepted_fullsix_control_metrics": EXPECTED_CONTROL_METRICS,
        "pilot_row_matched_control_metrics": EXPECTED_PILOT_CONTROL_METRICS,
        "gate": EXPECTED_GATE,
        "resource_policy": {"threads": THREADS, "concurrent_jobs": 1,
                            "job_memory_cap_bytes": JOB_CAP_BYTES,
                            "free_memory_reserve_bytes": FREE_RAM_RESERVE_BYTES},
        "test_data_used": False,
        "kaggle_access": False,
    })


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card_and_queue(queue, card)
    train_path, frame, target, cycles = r01._check_official_train()
    if file_sha256(train_path) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train hash differs after feature read")
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R16 STANDARD views changed: {views}")
    input_hashes = _validate_input_lineage()
    lineage = r01._validate_manifest(queue)
    controls = _load_control_arrays(frame, cycles, views)
    control_metrics = _baseline_metrics({"cycles": cycles, "views": views, "controls": controls}, target)
    memory = {
        str(cutoff): _memory_estimate(EXPECTED_FIT_ROWS[cutoff])
        for cutoff in (55, 67)
    }
    ledger = _read_budget_ledger()
    recipe = _recipe_payload(card, views)
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES, input_files=INPUTS,
        source_files=SOURCES, evaluation_views=views, packages=PACKAGES,
    )
    identity.update({
        "recipe": recipe,
        "control_lineage": lineage,
        "pilot_row_matched_control_metrics": control_metrics,
        "resource_policy": recipe["resource_policy"],
        "expected_fit_rows": EXPECTED_FIT_ROWS,
        "expected_score_rows": EXPECTED_VIEW_ROWS,
    })
    identity["recipe_artifact_sha256"] = json_sha256(recipe)
    signature = json_sha256(identity)
    if RECIPE_PATH.exists() and _read_json(RECIPE_PATH) != recipe:
        raise BatchIntegrityError("Existing R16 recipe differs from current canonical preflight recipe")
    if FREEZE_PATH.exists():
        frozen = _read_json(FREEZE_PATH)
        if frozen.get("identity") != identity or frozen.get("signature") != signature:
            raise BatchIntegrityError("Existing R16 freeze differs from current source/input/recipe identity")
    memory_pass = all(value.get("status") == "PASS" for value in memory.values())
    status = "PREFLIGHT_PASS" if memory_pass and ledger.get("status") == "PASS" else "BLOCKED_RESOURCE_OR_BUDGET"
    source_hashes = {str(Path(path).relative_to(ROOT)): file_sha256(path) for path in SOURCES}
    public = {
        "card_id": CARD_ID,
        "status": status,
        "scope": "official Regression Train and frozen control preflight; no new candidate prediction or fitting",
        "official_train_sha256": input_hashes["official_regression_train"],
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "freeze_identity_sha256": signature,
        "freeze_written": False,
        "recipe_payload": recipe,
        "control_lineage": lineage,
        "accepted_fullsix_control_metrics": EXPECTED_CONTROL_METRICS,
        "pilot_row_matched_control_metrics": control_metrics,
        "views": recipe["evaluation"]["views"],
        "fit_rows": EXPECTED_FIT_ROWS,
        "view_rows": EXPECTED_VIEW_ROWS,
        "control_row_id_hashes": {
            str(cutoff): hashlib.sha256("\n".join(map(str, value["score_record_ids"].tolist())).encode("utf-8")).hexdigest()
            for cutoff, value in controls.items()
        },
        "memory_estimate": memory,
        "budget_preflight": ledger,
        "training_lock": {"path": str(RUN_DIR.parent / ".training.lock"),
                          "acquired": False, "max_concurrent_jobs": 1},
        "candidate_prediction_or_metric_computed": False,
        "test_or_sample_access": False,
        "kaggle_access": False,
        "artifacts_written": False,
        "python_version": platform.python_version(),
    }
    return {"public": public, "identity": identity, "recipe": recipe,
            "frame": frame, "target": target, "cycles": cycles,
            "views": views, "controls": controls, "train_path": train_path}


def _control_loader(context: Mapping[str, Any]):
    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        saved = context["controls"].get(int(cutoff))
        if saved is None or not np.array_equal(saved["score_rows"], score_rows):
            raise BatchIntegrityError(f"R16 frozen control rows differ at cutoff {cutoff}")
        return {
            "baseline_prediction": saved["baseline_prediction"],
            "old_lgb_svd32": saved["old_lgb_svd32"],
            "ridge_prediction": saved["ridge_prediction"],
            "sparse_rank50": saved["sparse_rank50"],
            "control_rows": saved["score_rows"],
        }
    return load


def _resource_guarded_fit_predict(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Recheck the frozen memory bound immediately before each predictor fit."""
    resource = _memory_estimate(len(args[0]))
    if resource["status"] != "PASS":
        raise BatchIntegrityError(f"R16 fit memory gate is {resource['status']}")
    result = dict(r16_fit_predict(*args, **kwargs))
    result["fit_metadata"] = dict(result.get("fit_metadata", {}))
    result["fit_metadata"]["resource_preflight"] = resource
    return result


def _independent_rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    cycles = np.asarray(cycles, dtype=np.int64)
    if raw.ndim != 1 or raw.shape != cycles.shape or not np.isfinite(raw).all():
        raise BatchIntegrityError("R16 raw prediction/cycle arrays are invalid")
    ranked = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        selected = cycles == cycle
        ranked[selected] = (0.5 * raw[selected]
                            + 0.5 * 100.0 * rankdata(raw[selected], method="average") / int(selected.sum()))
    ranked = np.clip(ranked, 0.0, 100.0)
    if not np.isfinite(ranked).all():
        raise BatchIntegrityError("R16 rank50 component is nonfinite")
    return ranked


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim != 1 or target.shape != prediction.shape or not len(target):
        raise BatchIntegrityError("R16 MAE arrays are empty or misaligned")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise BatchIntegrityError("R16 MAE arrays are nonfinite")
    return float(np.mean(np.abs(target - prediction), dtype=np.float64))


def _fit_accounting(context: Mapping[str, Any]) -> dict[str, Any]:
    if not LEDGER_PATH.is_file():
        raise BatchIntegrityError("R16 fit ledger is missing")
    all_events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    events = [event for event in all_events if event.get("card_id") == CARD_ID]
    reservations = [event for event in events if event.get("event") == "BUDGET_RESERVED"]
    if (len(reservations) != 1 or reservations[0].get("budget_key") != "R16:PILOT"
            or int(reservations[0].get("predictor_fits", -1)) != 2):
        raise BatchIntegrityError("R16 ledger reservation is absent, duplicated or incorrect")
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event for event in events if event.get("event") == "FIT_SAVED"]
    reuses = [event for event in events if event.get("event") == "RESUMED_VERIFIED"]
    failures = [event for event in events if event.get("event") == "FIT_FAILED"]
    if not starts or any(int(event.get("cutoff", -1)) not in (55, 67)
                         for event in [*starts, *saves, *reuses, *failures]):
        raise BatchIntegrityError("R16 ledger is missing fit attempts or has an unknown cutoff")
    cutoffs: dict[str, Any] = {}
    for cutoff in (55, 67):
        path = RUN_DIR / f"R16_fit{cutoff}.npz"
        identity = {"freeze_signature": json_sha256(context["identity"]),
                    "card_id": CARD_ID, "cutoff": cutoff,
                    "score_rows": context["controls"][cutoff]["score_rows"].tolist()}
        _arrays, sidecar = read_bundle(path, identity)
        metadata = sidecar["metadata"]
        fit = metadata.get("fit_metadata", {})
        resource = fit.get("resource_preflight", {})
        cutoff_events = [event for event in events if int(event.get("cutoff", -1)) == cutoff]
        cutoff_starts = [event for event in cutoff_events if event.get("event") == "FIT_STARTED"]
        cutoff_saves = [event for event in cutoff_events if event.get("event") == "FIT_SAVED"]
        cutoff_reuses = [event for event in cutoff_events if event.get("event") == "RESUMED_VERIFIED"]
        cutoff_failures = [event for event in cutoff_events if event.get("event") == "FIT_FAILED"]
        if (len(cutoff_starts) < 1 or len(cutoff_saves) > 1 or len(cutoff_reuses) > 1
                or len(cutoff_saves) + len(cutoff_reuses) < 1
                or len(cutoff_failures) > len(cutoff_starts)):
            raise BatchIntegrityError(f"R16 cutoff {cutoff} attempt/save accounting is invalid")
        if (int(metadata.get("fit_predictor_count", -1)) != 1
                or int(metadata.get("fit_rows", -1)) != EXPECTED_FIT_ROWS[cutoff]
                or int(metadata.get("score_rows", -1)) != len(context["controls"][cutoff]["score_rows"])):
            raise BatchIntegrityError(f"R16 fit receipt does not reconcile at cutoff {cutoff}")
        if resource.get("status") != "PASS" or resource.get("concurrent_training_jobs") != 1:
            raise BatchIntegrityError(f"R16 resource/slot receipt is missing at cutoff {cutoff}")
        seconds = float(metadata.get("fit_seconds", float("nan")))
        if not np.isfinite(seconds) or seconds < 0:
            raise BatchIntegrityError(f"R16 fit time is invalid at cutoff {cutoff}")
        cutoffs[str(cutoff)] = {
            "predictor_fit_count": 1,
            "fit_rows": int(metadata["fit_rows"]),
            "score_rows": int(metadata["score_rows"]),
            "fit_seconds": seconds,
            "fit_attempt_count": len(cutoff_starts),
            "failed_attempt_count": len(cutoff_failures),
            "successful_cache_status": "FIT_SAVED" if cutoff_saves else "VERIFIED_REUSE",
            "attempt_events": [str(x.get("event")) for x in cutoff_events
                               if x.get("event") in {"FIT_STARTED", "FIT_SAVED", "FIT_FAILED", "RESUMED_VERIFIED"}],
            "training_warnings": fit.get("training_warnings", []),
            "resource_preflight": resource,
            "weight_audit": fit.get("sample_weighting", {}),
        }
    return {"budget_reserved_predictor_fits": 2, "successful_predictor_fits": 2,
            "cutoff_fits": cutoffs,
            "total_fit_seconds": float(sum(item["fit_seconds"] for item in cutoffs.values())),
            "peak_memory_observed": False}


def _pilot_group_sources(
    context: Mapping[str, Any],
    per_cutoff: Mapping[int, Mapping[str, np.ndarray]],
    view_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Build pilot, early-H12 and late-H12 sources from STANDARD fold caches."""
    sources = {name: dict(view_arrays[name]) for name in ("primary",)}
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    for name, cutoff, low_cycle, high_cycle in (
        ("H12_E", 55, 56, 67),
        ("H12_L", 67, 68, 79),
    ):
        fold = per_cutoff[cutoff]
        rows = np.asarray(fold["rows"], dtype=np.int64)
        selected = (cycles[rows] >= low_cycle) & (cycles[rows] <= high_cycle)
        group = {"rows": rows[selected],
                 "candidate": np.asarray(fold["candidate"], dtype=np.float64)[selected],
                 "baseline": np.asarray(fold["baseline"], dtype=np.float64)[selected]}
        if (len(group["rows"]) != EXPECTED_VIEW_ROWS[name]
                or len(np.unique(group["rows"])) != len(group["rows"])):
            raise BatchIntegrityError(f"R16 verifier denominator differs for {name}")
        sources[name] = group
    return sources


def _verify_pilot(context: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    if (report.get("card_id") != CARD_ID
            or report.get("status") != "PILOT_SAVED_AWAITING_INDEPENDENT_QA"
            or report.get("fit_predictor_count") != 2
            or report.get("cutoffs") != [55, 67]
            or report.get("freeze_signature") != json_sha256(context["identity"])):
        raise BatchIntegrityError("R16 pilot engine report identity is invalid")
    y = np.asarray(context["target"], dtype=np.float64)
    frame = context["frame"]
    cycles = np.asarray(context["cycles"], dtype=np.int64)
    record_ids = frame["record_id"].astype(str).to_numpy(dtype=str)
    per_cutoff: dict[int, dict[str, Any]] = {}
    for cutoff in (55, 67):
        rows = context["controls"][cutoff]["score_rows"]
        expected_identity = {"freeze_signature": json_sha256(context["identity"]),
                             "card_id": CARD_ID, "cutoff": cutoff,
                             "score_rows": rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"R16_fit{cutoff}.npz", expected_identity)
        if (not np.array_equal(arrays["score_rows"], rows)
                or not np.array_equal(arrays["score_record_ids"], record_ids[rows])
                or len(np.unique(arrays["score_record_ids"])) != len(rows)):
            raise BatchIntegrityError(f"R16 output row/ID alignment differs at cutoff {cutoff}")
        raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
        prediction = np.asarray(arrays["prediction"], dtype=np.float64)
        baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        old_lgb = context["controls"][cutoff]["old_lgb_svd32"]
        ridge = context["controls"][cutoff]["ridge_prediction"]
        sparse_rank = context["controls"][cutoff]["sparse_rank50"]
        expected_baseline = np.clip(0.5625 * old_lgb + 0.1875 * ridge + 0.25 * sparse_rank, 0, 100)
        rank = _independent_rank50(raw, cycles[rows])
        expected_prediction = np.clip(0.5625 * rank + 0.1875 * ridge + 0.25 * sparse_rank, 0, 100)
        if (not np.allclose(baseline, expected_baseline, rtol=0, atol=1e-10)
                or not np.allclose(prediction, expected_prediction, rtol=0, atol=1e-10)
                or not np.allclose(arrays["audit__candidate_rank50"], rank, rtol=0, atol=1e-10)
                or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))):
            raise BatchIntegrityError(f"R16 fixed route/rank mapping differs at cutoff {cutoff}")
        metadata = sidecar.get("metadata", {})
        fit = metadata.get("fit_metadata", {})
        model_state = metadata.get("state", {})
        fit_cycles = cycles[cycles <= cutoff]
        raw_weights = np.exp2(-(cutoff - fit_cycles).astype(np.float64) / 8.0)
        expected_weights = raw_weights / raw_weights.mean()
        weight_audit = fit.get("sample_weighting", {})
        model_weight_audit = model_state.get("sample_weighting", {})
        observed_weight_sha = hashlib.sha256(
            np.asarray(expected_weights, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        if (int(metadata.get("fit_rows", -1)) != len(fit_cycles)
                or int(metadata.get("fit_predictor_count", -1)) != 1
                or model_state.get("source_fit_cycles") != fit_cycles.astype(int).tolist()
                or model_weight_audit.get("sha256") != observed_weight_sha
                or int(model_state.get("cutoff", -1)) != cutoff
                or model_weight_audit.get("formula") != "raw_i=2**(-(cutoff-cycle_code_i)/8); normalized_i=raw_i/mean(raw)"):
            raise BatchIntegrityError(f"R16 weighted fit audit does not match fit rows at cutoff {cutoff}")
        if (abs(float(weight_audit.get("mean", float("nan"))) - 1.0) > 1e-12
                or int(weight_audit.get("fit_rows", -1)) != len(fit_cycles)
                or not np.isfinite(expected_weights).all()
                or np.any(expected_weights <= 0)):
            raise BatchIntegrityError(f"R16 sample-weight normalization is invalid at cutoff {cutoff}")
        per_cutoff[cutoff] = {"rows": rows, "candidate": prediction,
                              "baseline": baseline, "raw": raw, "rank50": rank}

    view_arrays: dict[str, dict[str, np.ndarray]] = {}
    for name, segments in context["views"].items():
        row_parts: list[np.ndarray] = []
        cand_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            origin = per_cutoff[cutoff]
            mask = (cycles[origin["rows"]] >= low) & (cycles[origin["rows"]] <= high)
            row_parts.append(origin["rows"][mask])
            cand_parts.append(origin["candidate"][mask])
            base_parts.append(origin["baseline"][mask])
        rows = np.concatenate(row_parts)
        if len(rows) != EXPECTED_VIEW_ROWS[name] or len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R16 verifier denominator differs for {name}")
        candidate = np.concatenate(cand_parts)
        baseline = np.concatenate(base_parts)
        cand_mae = _mae(y[rows], candidate)
        base_mae = _mae(y[rows], baseline)
        reported = report.get("views", {}).get(name, {})
        if (int(reported.get("rows", -1)) != len(rows)
                or abs(float(reported.get("candidate", {}).get("mae", float("nan"))) - cand_mae) > 1e-10
                or abs(float(reported.get("baseline", {}).get("mae", float("nan"))) - base_mae) > 1e-10):
            raise BatchIntegrityError(f"R16 engine and independent metrics differ for {name}")
        view_arrays[name] = {"rows": rows, "candidate": candidate, "baseline": baseline}

    groups: dict[str, Any] = {}
    group_sources = _pilot_group_sources(context, per_cutoff, view_arrays)
    stream_missing_all = frame["signal_stream"].fillna("").astype(str).str.strip().eq("").to_numpy()
    for view_name, source in group_sources.items():
        rows = source["rows"]
        local_y = y[rows]
        masks = {
            "stream_missing": stream_missing_all[rows],
            "target_le_5": local_y <= 5.0,
            "target_ge_95": local_y >= 95.0,
        }
        groups[view_name] = {}
        for group_name, mask in masks.items():
            count = int(mask.sum())
            if count < MIN_GROUP_SUPPORT:
                groups[view_name][group_name] = {"n": count, "supported": False,
                                                "candidate_mae": None, "baseline_mae": None,
                                                "harm": None}
                continue
            candidate_mae = _mae(local_y[mask], source["candidate"][mask])
            baseline_mae = _mae(local_y[mask], source["baseline"][mask])
            groups[view_name][group_name] = {
                "n": count, "supported": True,
                "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
                "harm": candidate_mae - baseline_mae,
            }

    views_result = {}
    for name, source in view_arrays.items():
        cand_mae = _mae(y[source["rows"]], source["candidate"])
        base_mae = _mae(y[source["rows"]], source["baseline"])
        views_result[name] = {"rows": len(source["rows"]), "candidate_mae": cand_mae,
                              "baseline_mae": base_mae,
                              "gain_positive_is_better": base_mae - cand_mae}
    gate_checks = {
        "primary_gain_at_least_0_03": views_result["primary"]["gain_positive_is_better"] >= EXPECTED_GATE["pilot_primary_MAE_gain_min"],
        "H12_harm_at_most_0_05": views_result["H12"]["candidate_mae"] - views_result["H12"]["baseline_mae"] <= EXPECTED_GATE["pilot_H12_MAE_harm_max"],
        "each_pilot_window_harm_at_most_0_5": all(
            views_result[name]["candidate_mae"] - views_result[name]["baseline_mae"]
            <= EXPECTED_GATE["individual_pilot_view_MAE_harm_max"] for name in ("W1", "W4")),
    }
    supported_harms = [item["harm"] for view in groups.values() for item in view.values()
                       if item["supported"]]
    max_group_harm = max(supported_harms) if supported_harms else None
    gate_checks["supported_tail_missing_harm_at_most_0_25"] = (
        max_group_harm is not None
        and max_group_harm <= EXPECTED_GATE["supported_tail_or_missing_MAE_harm_max"])
    decision = "ADVANCE" if all(gate_checks.values()) else "RETIRE"
    return {
        "card_id": CARD_ID,
        "status": "INDEPENDENT_QA_PASS",
        "decision": decision,
        "freeze_signature": json_sha256(context["identity"]),
        "fit_predictor_count": 2,
        "cutoffs": [55, 67],
        "views": views_result,
        "groups": groups,
        "maximum_supported_group_harm": max_group_harm,
        "gate_checks": gate_checks,
        "gate_definition": EXPECTED_GATE,
        "test_data_used": False,
        "kaggle_access": False,
    }


def _publish(context: Mapping[str, Any], report: Mapping[str, Any],
             verification: Mapping[str, Any], accounting: Mapping[str, Any]) -> None:
    gate = {"card_id": CARD_ID, "decision": verification["decision"],
            "checks": verification["gate_checks"], "thresholds": EXPECTED_GATE,
            "status": "PASS" if verification["decision"] == "ADVANCE" else "FAIL_VALID_EVIDENCE"}
    result = dict(report)
    result.update({
        "status": "PILOT_COMPLETE_" + verification["decision"],
        "input_sha256": context["public"]["input_sha256"],
        "source_sha256": context["public"]["source_sha256"],
        "preflight": context["public"],
        "fit_accounting": dict(accounting),
        "independent_verification": {key: verification[key] for key in (
            "status", "decision", "views", "groups", "maximum_supported_group_harm", "gate_checks")},
        "resource_note": "RAM values are conservative prefit estimates; process peak RAM was not observed.",
        "test_data_used": False,
        "kaggle_access": False,
    })
    summary = {
        "card_id": CARD_ID,
        "decision": verification["decision"],
        "fit_predictor_count": 2,
        "cutoffs": [55, 67],
        "primary_rows": EXPECTED_VIEW_ROWS["primary"],
        "primary_baseline_mae": verification["views"]["primary"]["baseline_mae"],
        "primary_candidate_mae": verification["views"]["primary"]["candidate_mae"],
        "primary_gain": verification["views"]["primary"]["gain_positive_is_better"],
        "H12_rows": EXPECTED_VIEW_ROWS["H12"],
        "H12_baseline_mae": verification["views"]["H12"]["baseline_mae"],
        "H12_candidate_mae": verification["views"]["H12"]["candidate_mae"],
        "H12_gain": verification["views"]["H12"]["gain_positive_is_better"],
        "total_fit_seconds": accounting["total_fit_seconds"],
        "test_sample_kaggle_used": False,
        "details": ["results.json", "verification.json", "gate.json"],
    }
    _write_or_match(RESULTS_PATH, result)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(SUMMARY_PATH, summary)


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one R16 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError("R16 is blocked by frozen input, resource, or fit-budget preflight")
    if freeze_only:
        if RECIPE_PATH.exists():
            if _read_json(RECIPE_PATH) != context["recipe"]:
                raise BatchIntegrityError("Existing R16 recipe differs; refusing rewrite")
        else:
            write_json_exclusive(RECIPE_PATH, context["recipe"])
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8")) if FREEZE_PATH.exists() else None
        identity = dict(context["identity"])
        if frozen is None:
            from src.models.optimization_v3_batch1 import ensure_frozen
            ensure_frozen(FREEZE_PATH, identity)
        elif frozen.get("identity") != identity or frozen.get("signature") != json_sha256(identity):
            raise BatchIntegrityError("Existing R16 freeze differs; refusing overwrite")
        result = dict(context["public"])
        result.update({"status": "FROZEN_READY", "freeze_written": True,
                       "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
                       "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                       "freeze_sha256": file_sha256(FREEZE_PATH),
                       "candidate_prediction_or_metric_computed": False,
                       "artifacts_written": True})
        return result

    if not FREEZE_PATH.is_file() or not RECIPE_PATH.is_file():
        raise BatchIntegrityError("R16 run requires a committed recipe and prefit freeze")
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train changed after R16 freeze")
    target_frame = pd.read_csv(context["train_path"], usecols=["threat_rank"])
    if file_sha256(context["train_path"]) != TRAIN_SHA256:
        raise BatchIntegrityError("Official Train changed while target was read")
    target = pd.to_numeric(target_frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (target.shape != (len(context["frame"]),) or not np.isfinite(target).all()
            or np.any((target < 0) | (target > 100))):
        raise BatchIntegrityError("Official Regression target is invalid or misaligned")
    frame = context["frame"].copy()
    frame["threat_rank"] = target
    report = run_pilot(
        card=card,
        artifact_dir=RUN_DIR,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=MAX_SUCCESSFUL_FITS,
        frame=frame,
        target=target,
        feature_names=FEATURE_NAMES,
        adapter=_resource_guarded_fit_predict,
        control_loader=_control_loader(context),
        freeze_identity=context["identity"],
        freeze_path=FREEZE_PATH,
        event_path=LEDGER_PATH,
        evaluation_views=context["views"],
        threads=THREADS,
        report_path=ENGINE_REPORT_PATH,
    )
    verification = _verify_pilot({**context, "target": target}, report)
    accounting = _fit_accounting(context)
    _publish(context, report, verification, accounting)
    return {"card_id": CARD_ID, "status": "PILOT_COMPLETE_" + verification["decision"],
            "decision": verification["decision"], "views": verification["views"],
            "gate_checks": verification["gate_checks"], "fit_accounting": accounting,
            "freeze_signature": json_sha256(context["identity"]),
            "test_data_used": False, "kaggle_access": False}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(_json_clean(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
